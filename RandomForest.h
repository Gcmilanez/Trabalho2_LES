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

    // Métodos de Treino [MANTIDOS]
    void fit_baseline(const std::vector<std::vector<double>>& X, const std::vector<int>& y);
    void fit_optimized(const std::vector<std::vector<double>>& X, const std::vector<int>& y);

    // Predição Original (Compatibilidade)
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // [NOVO] Predição Otimizada para Dados Flat (Linear Memory Layout)
    std::vector<int> predict_flat(const std::vector<double>& X_flat, int n_samples, int n_features) const;
    
    // Serialização [MANTIDO]
    void save_model(const std::string& filename) const;
    void load_model(const std::string& filename);
    
    int get_num_trees() const { return n_trees; }

private:
    int n_trees;
    int max_depth;
    int min_samples_split;
    int chunk_size;

    std::vector<DecisionTree> trees;
    
    // Métodos auxiliares [MANTIDOS]
    void bootstrap_indices(int n_samples, std::vector<int>& out) const; 
    int majority_vote(const int* votes, int size) const;
    int majority_vote(const std::vector<int>& votes) const;
};

#endif // RANDOM_FOREST_H