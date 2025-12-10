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
// FIT BASELINE (Memória Padrão, Bootstrap Padrão)
// ============================================================
void RandomForest::fit_baseline(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y)
{
    trees.clear(); trees.reserve(n_trees);
    int n_samples = X.size();
    std::vector<int> indices; 
    indices.reserve(n_samples);

    for (int t = 0; t < n_trees; t++) {
        bootstrap_indices(n_samples, indices); // Amostragem com reposição
        
        DecisionTree dt(max_depth, min_samples_split, chunk_size);
        dt.fit_baseline(X, y, indices); // Chama método Baseline
        
        trees.push_back(std::move(dt));
    }
}

// ============================================================
// FIT OTIMIZADO (Global Flat Memory, Cache Indices)
// ============================================================
void RandomForest::fit_optimized(const std::vector<std::vector<double>>& X,
                                 const std::vector<int>& y)
{
    trees.clear(); trees.reserve(n_trees);
    int n_samples = X.size();
    int n_features = X[0].size();

    // 1. Preparação Global da Memória (Flattening)
    std::vector<double> X_flat_global(n_samples * n_features);
    for (int f = 0; f < n_features; f++) {
        for (int i = 0; i < n_samples; i++) {
            X_flat_global[f * n_samples + i] = X[i][f];
        }
    }

    // 2. Preparação de Índices
    init_base_indices(n_samples);
    std::vector<int> indices;
    indices.reserve(n_samples);

    for (int t = 0; t < n_trees; t++) {
        make_cache_friendly_indices(n_samples, t, indices);

        DecisionTree dt(max_depth, min_samples_split, chunk_size);
        dt.fit_optimized(X_flat_global, n_samples, n_features, y, indices); // Chama método Otimizado
        
        trees.push_back(std::move(dt));
    }
}

// Auxiliares (bootstrap, init_base...) e Serialização padrão
void RandomForest::bootstrap_indices(int n_samples, std::vector<int>& out) const {
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n_samples - 1);
    out.clear();
    for(int i=0; i<n_samples; ++i) out.push_back(dist(gen));
}

void RandomForest::init_base_indices(int n_samples) {
    base_indices.resize(n_samples);
    std::iota(base_indices.begin(), base_indices.end(), 0);
    static thread_local std::mt19937 gen(std::random_device{}());
    std::shuffle(base_indices.begin(), base_indices.end(), gen);
}

void RandomForest::make_cache_friendly_indices(int n_samples, int tree_id, std::vector<int>& out) const {
    out.resize(n_samples);
    int offset = (tree_id * chunk_size) % n_samples;
    for(int i=0; i<n_samples; ++i) {
        out[i] = base_indices[(i + offset) % n_samples];
    }
}

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