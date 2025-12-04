#ifndef RANDOM_FOREST_BASELINE_H
#define RANDOM_FOREST_BASELINE_H

#include <vector>
#include <string>
#include "DecisionTree.h"

// ------------------------------------------------------------
// RandomForestBaseline
// Implementação simples de Random Forest sem otimizações
// ------------------------------------------------------------
class RandomForestBaseline {
public:
    RandomForestBaseline(int n_trees = 10,
                         int max_depth = 10,
                         int min_samples_split = 2);

    // Treino da floresta
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);

    // Predição em várias amostras
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // --------------------------------------------------------
    // Serialização binária (modelo completo da floresta)
    // --------------------------------------------------------
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);

    // Getters úteis
    int get_num_trees() const          { return n_trees; }
    int get_max_depth() const          { return max_depth; }
    int get_min_samples_split() const  { return min_samples_split; }

private:
    int n_trees;
    int max_depth;
    int min_samples_split;

    std::vector<DecisionTree> trees;

    // Buffers para votação (evita realocação)
    mutable std::vector<int> vote_buffer;

    // Auxiliares
    void bootstrap_indices(int n_samples, std::vector<int>& out) const;
    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_BASELINE_H
