#include "RandomForest.h"
#include <fstream>
#include <numeric>
#include <algorithm>
#include <unordered_map>
#include <iostream>

RandomForest::RandomForest(int n_trees, int max_depth, int min_samples_split, int chunk_size)
    : n_trees(n_trees), max_depth(max_depth), min_samples_split(min_samples_split), chunk_size(chunk_size)
{
    trees.reserve(n_trees);
}

// ============================================================
// FIT BASELINE (MANTIDO INTOCADO - PADRÃO SKLEARN LENTO)
// ============================================================
void RandomForest::fit_baseline(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y)
{
    trees.clear(); trees.reserve(n_trees);
    int n_samples = X.size();
    if (n_samples == 0) return;
    // O Baseline não faz flattening (usa vector<vector> direto) na árvore, 
    // mas se a assinatura da árvore pede flat, nós convertemos de forma simples aqui.
    // Assumindo que o DecisionTree.h pede flat para ambos (conforme último passo):
    
    int n_features = X[0].size();
    std::vector<double> X_flat_global(n_samples * n_features);
    // Flattening Naive (Coluna por Coluna - Cache Hostile na leitura de X)
    for (int f = 0; f < n_features; f++) {
        for (int i = 0; i < n_samples; i++) {
            X_flat_global[f * n_samples + i] = X[i][f];
        }
    }

    std::vector<int> indices; 
    indices.reserve(n_samples);

    for (int t = 0; t < n_trees; t++) {
        bootstrap_indices(n_samples, indices); // Gera índices aleatórios desordenados
        
        DecisionTree dt(max_depth, min_samples_split, chunk_size);
        dt.fit_baseline(X_flat_global, n_samples, n_features, y, indices); 
        trees.push_back(std::move(dt));
    }
}

// ============================================================
// FIT OTIMIZADO (CACHE IMPROVEMENTS)
// ============================================================
void RandomForest::fit_optimized(const std::vector<std::vector<double>>& X,
                                 const std::vector<int>& y)
{
    trees.clear(); trees.reserve(n_trees);
    int n_samples = X.size();
    if (n_samples == 0) return;
    int n_features = X[0].size();

    // 1. FLATTENING OTIMIZADO (CACHE BLOCKING / TILING)
    // Transpõe a matriz usando blocos para manter os dados no Cache L1/L2
    // Evita cache trashing ao ler X row-major e escrever X_flat column-major
    std::vector<double> X_flat_global(n_samples * n_features);
    
    const int BLOCK_SIZE = 32; // 32x32 doubles = 8KB (Cabe no L1)

    for (int i = 0; i < n_samples; i += BLOCK_SIZE) {
        for (int f = 0; f < n_features; f += BLOCK_SIZE) {
            int i_max = std::min(i + BLOCK_SIZE, n_samples);
            int f_max = std::min(f + BLOCK_SIZE, n_features);

            for (int ii = i; ii < i_max; ++ii) {
                for (int ff = f; ff < f_max; ++ff) {
                    X_flat_global[ff * n_samples + ii] = X[ii][ff];
                }
            }
        }
    }

    std::vector<int> indices;
    indices.reserve(n_samples);

    for (int t = 0; t < n_trees; t++) {
        // 2. BOOTSTRAP OTIMIZADO (MONOTONIC ACCESS)
        // Passo A: Gera aleatoriedade estatística (igual Sklearn)
        bootstrap_indices(n_samples, indices);
        
        // Passo B: ORDENA os índices!
        // Isso garante que a leitura de memória dentro da árvore seja sempre
        // crescente (0, 2, 5, 5, 9...), ativando o Hardware Prefetcher.
        // O resultado da árvore é idêntico (ordem não importa p/ soma), mas é muito mais rápido.
        std::sort(indices.begin(), indices.end());

        DecisionTree dt(max_depth, min_samples_split, chunk_size);
        dt.fit_optimized(X_flat_global, n_samples, n_features, y, indices);
        trees.push_back(std::move(dt));
    }
}

// ============================================================
// AUXILIARES
// ============================================================
void RandomForest::bootstrap_indices(int n_samples, std::vector<int>& out) const {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n_samples - 1);
    out.clear();
    for(int i=0; i<n_samples; ++i) out.push_back(dist(gen));
}

// (init_base_indices e make_cache_friendly_indices foram removidos 
// pois agora usamos Bootstrap+Sort para máxima compatibilidade e performance)

std::vector<int> RandomForest::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    vote_buffer.resize(n_trees);
    for (const auto& sample : X) {
        for (int t = 0; t < n_trees; t++) {
            vote_buffer[t] = trees[t].predict_one(sample);
        }
        predictions.push_back(majority_vote(vote_buffer));
    }
    return predictions;
}

int RandomForest::majority_vote(const std::vector<int>& votes) const {
    std::unordered_map<int, int> freq;
    for (int v : votes) freq[v]++;
    int best_class = -1, best_count = -1;
    for (auto& kv : freq) {
        if (kv.second > best_count) {
            best_count = kv.second;
            best_class = kv.first;
        }
    }
    return best_class;
}

void RandomForest::save_model(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary);
    if (!out) throw std::runtime_error("Erro ao salvar modelo: " + filename);
    out.write((char*)&n_trees, sizeof(int));
    out.write((char*)&max_depth, sizeof(int));
    out.write((char*)&min_samples_split, sizeof(int));
    out.write((char*)&chunk_size, sizeof(int));
    for (const auto& tree : trees) tree.save_model(out);
}

void RandomForest::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary);
    if (!in) throw std::runtime_error("Erro ao ler modelo: " + filename);
    in.read((char*)&n_trees, sizeof(int));
    in.read((char*)&max_depth, sizeof(int));
    in.read((char*)&min_samples_split, sizeof(int));
    in.read((char*)&chunk_size, sizeof(int));
    trees.clear();
    trees.reserve(n_trees);
    for (int t = 0; t < n_trees; t++) {
        DecisionTree tree(max_depth, min_samples_split, chunk_size);
        tree.load_model(in);
        trees.emplace_back(std::move(tree));
    }
}