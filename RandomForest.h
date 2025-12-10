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

    void fit_baseline(const std::vector<std::vector<double>>& X,
                      const std::vector<int>& y);

    void fit_optimized(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
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
    
    void bootstrap_indices(int n_samples, std::vector<int>& out) const; 
    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_H