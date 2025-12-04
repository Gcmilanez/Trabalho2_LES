#ifndef RANDOM_FOREST_OPTIMIZED_H
#define RANDOM_FOREST_OPTIMIZED_H

#include <vector>
#include <string>
#include "DecisionTree.h"

// ------------------------------------------------------------
// RandomForestOptimized
// Versão da floresta que usa DecisionTree com chunks
// ------------------------------------------------------------
class RandomForestOptimized {
public:
    RandomForestOptimized(int n_trees = 10,
                          int max_depth = 10,
                          int min_samples_split = 2,
                          int chunk_size = DecisionTree::DEFAULT_CHUNK_SIZE);

    // Treino da floresta com processamento em chunks
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);

    // Predição
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // Serialização binária do modelo inteiro
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

    // Getters
    int get_num_trees() const          { return n_trees; }
    int get_max_depth() const          { return max_depth; }
    int get_min_samples_split() const  { return min_samples_split; }
    int get_chunk_size() const         { return chunk_size; }

private:
    int n_trees;
    int max_depth;
    int min_samples_split;
    int chunk_size;

    std::vector<DecisionTree> trees;

    // Buffers auxiliares
    std::vector<int> base_indices;
    std::vector<int> temp_indices;
    mutable std::vector<int> vote_buffer;

    // Auxiliares internos
    void init_base_indices(int n_samples);
    void make_cache_friendly_indices(int n_samples,
                                     int tree_id,
                                     std::vector<int>& out_indices) const;

    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_OPTIMIZED_H
