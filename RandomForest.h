#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <string>
#include <random>
#include "DecisionTree.h" // Sua classe DecisionTree (que já suporta chunks)

class RandomForest {
public:
    // Construtor com valores padrão baseados na sua imagem
    RandomForest(int n_trees = 50,
                 int max_depth = 8,
                 int min_samples_split = 5,
                 int chunk_size = 256);

    // --- TREINO UNIFICADO ---
    // use_optimized = false -> Baseline (Bootstrap Padrão + DecisionTree Basic)
    // use_optimized = true  -> Otimizado (Cache-Friendly Indices + DecisionTree Chunked)
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             bool use_optimized);

    // Predição
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // Serialização
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

    // Getters
    int get_num_trees() const { return n_trees; }

private:
    int n_trees;
    int max_depth;
    int min_samples_split;
    int chunk_size;

    std::vector<DecisionTree> trees;

    // Buffers e Auxiliares
    mutable std::vector<int> vote_buffer;
    
    // Auxiliares Otimizados
    std::vector<int> base_indices; // Usado apenas no modo otimizado
    
    // Métodos Internos
    void bootstrap_indices(int n_samples, std::vector<int>& out) const; // Baseline
    void init_base_indices(int n_samples); // Otimizado
    void make_cache_friendly_indices(int n_samples, int tree_id, std::vector<int>& out) const; // Otimizado
    
    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_H