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

// [Manter fit_baseline, fit_optimized e bootstrap_indices iguais]
void RandomForest::fit_baseline(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    trees.clear(); trees.reserve(n_trees); int n_samples = X.size(); if(n_samples==0) return; int n_features = X[0].size();
    std::vector<double> X_flat_global(n_samples * n_features);
    for (int f=0; f<n_features; f++) for (int i=0; i<n_samples; i++) X_flat_global[f*n_samples+i] = X[i][f];
    std::vector<int> indices; indices.reserve(n_samples);
    for (int t=0; t<n_trees; t++) {
        bootstrap_indices(n_samples, indices);
        DecisionTree dt(max_depth, min_samples_split, chunk_size);
        dt.fit_baseline(X_flat_global, n_samples, n_features, y, indices);
        trees.push_back(std::move(dt));
    }
}
void RandomForest::fit_optimized(const std::vector<std::vector<double>>& X, const std::vector<int>& y) {
    trees.clear(); trees.reserve(n_trees); int n_samples = X.size(); if(n_samples==0) return; int n_features = X[0].size();
    std::vector<double> X_flat_global(n_samples * n_features); const int BLOCK_SIZE = 32;
    for (int i=0; i<n_samples; i+=BLOCK_SIZE) for (int f=0; f<n_features; f+=BLOCK_SIZE) {
        int imax=std::min(i+BLOCK_SIZE, n_samples), fmax=std::min(f+BLOCK_SIZE, n_features);
        for(int k=i; k<imax; ++k) for(int l=f; l<fmax; ++l) X_flat_global[l*n_samples+k] = X[k][l];
    }
    std::vector<int> indices; indices.reserve(n_samples);
    for (int t=0; t<n_trees; t++) {
        bootstrap_indices(n_samples, indices);
        DecisionTree dt(max_depth, min_samples_split, chunk_size);
        dt.fit_optimized(X_flat_global, n_samples, n_features, y, indices);
        trees.push_back(std::move(dt));
    }
}
void RandomForest::bootstrap_indices(int n_samples, std::vector<int>& out) const {
    static thread_local std::mt19937 gen(std::random_device{}()); std::uniform_int_distribution<int> dist(0, n_samples - 1);
    out.clear(); for(int i=0; i<n_samples; ++i) out.push_back(dist(gen));
}


std::vector<int> RandomForest::predict_flat(const std::vector<double>& X_flat, int n_samples, int n_features) const {
    if (n_samples == 0) return {};
    std::vector<int> predictions(n_samples);

    // 1. AJUSTE DE CACHE L1:
    // O Batch precisa caber no Cache L1 (aprox 32KB). 
    // Uma amostra Covertype (54 feats) = 432 bytes. 
    // 128 amostras * 432 = 55KB (Cabe no L2 folgado, transborda pouco do L1).
    // 64 amostras * 432 = 27KB (Cabe INTEIRO no L1).
    const size_t BATCH_SIZE = 64; 
    
    // Buffer Transposto: [Tree][Sample]
    // Isso permite escrita sequencial no loop mais interno (Otimização de Store Buffer)
    std::vector<int> votes_buffer(n_trees * BATCH_SIZE);

    for (size_t start = 0; start < (size_t)n_samples; start += BATCH_SIZE) {
        size_t end = std::min(start + BATCH_SIZE, (size_t)n_samples);
        size_t current_batch_size = end - start;

        // Limpar buffer não é estritamente necessário se preenchermos tudo, 
        // mas é bom por segurança se a lógica mudar. Aqui sobrescrevemos, então ok.

        // --- FASE 1: PREDIÇÃO (Heavy CPU) ---
        // Loop Externo: Árvores (Mantém estrutura da árvore no Cache)
        for (int t = 0; t < n_trees; ++t) {
            const auto& tree = trees[t];
            
            // Ponteiro para a linha desta árvore no buffer
            int* tree_votes_ptr = &votes_buffer[t * BATCH_SIZE]; // Acesso [Tree][Sample]

            // Loop Interno: Amostras do Batch
            for (size_t i = 0; i < current_batch_size; ++i) {
                // Acesso Linear à Memória da Amostra (Graças ao Batch pequeno, isso está quente no L1/L2)
                const double* sample_ptr = &X_flat[(start + i) * n_features];
                
                // Predição Rápida
                int pred = tree.predict_sample_flat_ptr(sample_ptr);
                
                // ESCRITA LINEAR (Muito mais rápido que pular n_trees posições)
                tree_votes_ptr[i] = pred;
            }
        }

        // --- FASE 2: VOTAÇÃO (Memory Bound, mas leve) ---
        // Agora precisamos ler "colunas" do votes_buffer (Strided Read), 
        // mas como são inteiros e estão em cache L2, é rápido.
        
        // Pequena otimização: vetor local para contagem para evitar alocação de map repetida
        // Assumindo < 20 classes para Covertype/Digits. Se for genérico, map é seguro.
        // Vamos usar o método majority_vote genérico, mas passando pointers com stride.
        
        std::vector<int> sample_votes(n_trees); // Reutilizável

        for (size_t i = 0; i < current_batch_size; ++i) {
            // Coletar votos para a amostra 'i' de todas as árvores
            for (int t = 0; t < n_trees; ++t) {
                // Leitura com salto (Stride), mas inevitável sem transpor dados originais
                sample_votes[t] = votes_buffer[t * BATCH_SIZE + i];
            }
            predictions[start + i] = majority_vote(sample_votes);
        }
    }

    return predictions;
}

// ============================================================
// PREDIÇÃO OTIMIZADA COM BATCHING (TILING)
// ============================================================
std::vector<int> RandomForest::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    // Row-Major clássico
    for (const auto& sample : X) {
        std::vector<int> votes;
        votes.reserve(n_trees);
        for (const auto& tree : trees) {
            votes.push_back(tree.predict_one(sample));
        }
        predictions.push_back(majority_vote(votes));
    }
    return predictions;
}

// [Manter implementações de majority_vote, save_model e load_model]
int RandomForest::majority_vote(const int* votes, int size) const {
    if (size <= 0) return -1;
    // Otimização para pequenos loops: array fixo se soubermos max classes, 
    // senão map é o mais seguro.
    std::unordered_map<int, int> freq;
    int best_class = -1;
    int best_count = -1;
    for (int i = 0; i < size; ++i) {
        int v = votes[i];
        int count = ++freq[v];
        if (count > best_count) {
            best_count = count;
            best_class = v;
        }
    }
    return best_class;
}

int RandomForest::majority_vote(const std::vector<int>& votes) const {
    return majority_vote(votes.data(), votes.size());
}

void RandomForest::save_model(const std::string& filename) const {
    std::ofstream out(filename, std::ios::binary); if(!out) throw std::runtime_error("Erro salvar");
    out.write((char*)&n_trees, sizeof(int)); out.write((char*)&max_depth, sizeof(int));
    out.write((char*)&min_samples_split, sizeof(int)); out.write((char*)&chunk_size, sizeof(int));
    for (const auto& tree : trees) tree.save_model(out);
}
void RandomForest::load_model(const std::string& filename) {
    std::ifstream in(filename, std::ios::binary); if(!in) throw std::runtime_error("Erro ler");
    in.read((char*)&n_trees, sizeof(int)); in.read((char*)&max_depth, sizeof(int));
    in.read((char*)&min_samples_split, sizeof(int)); in.read((char*)&chunk_size, sizeof(int));
    trees.clear(); trees.reserve(n_trees);
    for (int t=0; t<n_trees; t++) {
        DecisionTree tree(max_depth, min_samples_split, chunk_size);
        tree.load_model(in);
        trees.emplace_back(std::move(tree));
    }
}