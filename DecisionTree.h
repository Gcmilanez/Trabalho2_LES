#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>
#include <utility>
#include <iostream>

struct Node {
    bool is_leaf = false;
    int feature_index = -1;
    double threshold = 0.0;
    int predicted_class = -1;

    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

class DecisionTree {
public:
    static constexpr int DEFAULT_CHUNK_SIZE = 256;

    DecisionTree(int max_depth = 10, 
                 int min_samples_split = 2, 
                 int chunk_size = DEFAULT_CHUNK_SIZE);

    // Movimentação
    DecisionTree(DecisionTree&&) noexcept = default;
    DecisionTree& operator=(DecisionTree&&) noexcept = default;
    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;

    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             bool use_optimized,
             const std::vector<int>* bootstrap_indices = nullptr);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    int predict_one(const std::vector<double>& sample) const;

    void save_model(std::ostream& out) const;
    void load_model(std::istream& in);

private:
    std::unique_ptr<Node> root;
    int max_depth;
    int min_samples_split;
    int chunk_size;
    bool use_optimized_mode;

    // --- ESTRUTURAS DE HISTOGRAMA (Novo) ---
    // Mapeia o índice do bin (0-255) de volta para o valor real (double)
    // bin_thresholds[feature_idx][bin_idx]
    std::vector<std::vector<double>> bin_thresholds;

    // Métodos Internos
    std::unique_ptr<Node> build_tree(const std::vector<std::vector<double>>* X_naive,
                                     const std::vector<uint8_t>* X_binned, // Usa uint8 agora!
                                     int n_total_samples,
                                     int n_features,
                                     const std::vector<int>& y,
                                     const std::vector<int>& indices,
                                     int depth);

    // BASELINE (Igual ao Sklearn: O(N log N))
    void find_best_split_naive(const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y,
                               const std::vector<int>& indices,
                               int& best_feature,
                               double& best_threshold,
                               std::vector<int>& left_idx,
                               std::vector<int>& right_idx,
                               double parent_gini);

    // OPTIMIZED (HISTOGRAMAS: O(N))
    void find_best_split_histogram(const std::vector<uint8_t>& X_binned,
                                   int n_total_samples,
                                   int n_features,
                                   const std::vector<int>& y,
                                   const std::vector<int>& indices,
                                   int& best_feature,
                                   double& best_threshold,
                                   std::vector<int>& left_idx,
                                   std::vector<int>& right_idx,
                                   double parent_gini);

    // Helpers
    double calculate_gini(const std::vector<int>& labels) const;
    int majority_class(const std::vector<int>& labels) const;
    int predict_sample(const std::vector<double>& sample, const Node* node) const;
    
    // Auxiliar para criar os bins
    void discretize_features(const std::vector<std::vector<double>>& X,
                             std::vector<uint8_t>& X_binned,
                             std::vector<std::vector<double>>& thresholds);

    void save_node(std::ostream& out, const Node* node) const;
    std::unique_ptr<Node> load_node(std::istream& in);
};

#endif // DECISION_TREE_H