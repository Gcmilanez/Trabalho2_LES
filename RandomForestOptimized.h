// RandomForestOptimized.h
#ifndef RANDOM_FOREST_OPTIMIZED_H
#define RANDOM_FOREST_OPTIMIZED_H

#include <vector>
#include "DecisionTree.h"

class RandomForestOptimized {
public:
    RandomForestOptimized(
        int n_trees = 32,
        int max_depth = 10,
        int min_samples_split = 2,
        int chunk_size = 100
        // aqui depois podemos adicionar: min_gain, n_thresholds_sample, etc.
    );

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    int predict_one(const std::vector<double>& x) const;

private:
    int n_trees;
    int max_depth;
    int min_samples_split;
    int chunk_size;

    std::vector<DecisionTree> trees;

    // buffers reutilizáveis
    mutable std::vector<int> vote_buffer;
    std::vector<int> base_indices;  // para bootstrap cache-friendly
    std::vector<int> indices_buffer; // para subconjuntos por árvore

    void init_base_indices(int n_samples);
    void make_cache_friendly_indices(int n_samples, int tree_id,
                                     std::vector<int>& out_indices) const;

    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_OPTIMIZED_H
