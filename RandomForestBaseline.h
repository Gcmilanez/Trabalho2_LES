// RandomForestBaseline.h
#ifndef RANDOM_FOREST_BASELINE_H
#define RANDOM_FOREST_BASELINE_H

#include <vector>
#include "DecisionTree.h"

class RandomForestBaseline {
public:
    RandomForestBaseline(
        int n_trees = 50,
        int max_depth = 10,
        int min_samples_split = 2
        // se sua DecisionTree tiver mais parâmetros, adicione aqui
    );

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // opcional: atalho para uma amostra só
    int predict_one(const std::vector<double>& x) const;

private:
    int n_trees;
    int max_depth;
    int min_samples_split;

    std::vector<DecisionTree> trees;

    // buffer reutilizável para votos (evita alocações repetidas)
    mutable std::vector<int> vote_buffer;

    void bootstrap_indices(int n_samples, std::vector<int>& indices) const;
    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_BASELINE_H
