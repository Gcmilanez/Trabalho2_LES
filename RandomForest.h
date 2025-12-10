#ifndef RANDOM_FOREST_H
#define RANDOM_FOREST_H

#include <vector>
#include <string>
#include <random>
#include "DecisionTree.h"

class RandomForest {
public:
    RandomForest(int n_trees = 50,
                 int max_depth = 8,
                 int min_samples_split = 5,
                 int chunk_size = 256);

    // --- MUDANÇA AQUI: Métodos Separados ---
    
    // Método 1: Baseline (Recebe dados brutos e treina lento)
    void fit_baseline(const std::vector<std::vector<double>>& X,
                      const std::vector<int>& y);

    // Método 2: Otimizado (Faz o Flattening Global e treina rápido)
    void fit_optimized(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y);

    // Predição (igual para ambos)
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // Serialização
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

    int get_num_trees() const { return n_trees; }

private:
    int n_trees;
    int max_depth;
    int min_samples_split;
    int chunk_size;

    std::vector<DecisionTree> trees;
    mutable std::vector<int> vote_buffer;
    std::vector<int> base_indices; 
    
    // Helpers internos
    void bootstrap_indices(int n_samples, std::vector<int>& out) const; 
    void init_base_indices(int n_samples); 
    void make_cache_friendly_indices(int n_samples, int tree_id, std::vector<int>& out) const; 
    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_H