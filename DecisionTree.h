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
    static constexpr int DEFAULT_CHUNK_SIZE = 100;

    DecisionTree(int max_depth = 10, 
                 int min_samples_split = 2, 
                 int chunk_size = DEFAULT_CHUNK_SIZE);

    // Construtor Padrão
    DecisionTree() : DecisionTree(10, 2, DEFAULT_CHUNK_SIZE) {}

    // Movimentação
    DecisionTree(DecisionTree&&) noexcept = default;
    DecisionTree& operator=(DecisionTree&&) noexcept = default;
    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;

    // --- ENTRADA 1: BASELINE (Lento / Padrão) ---
    void fit_baseline(const std::vector<std::vector<double>>& X,
                      const std::vector<int>& y,
                      const std::vector<int>& indices);

    // --- ENTRADA 2: OTIMIZADO (Rápido / Flat) ---
    void fit_optimized(const std::vector<double>& X_flat,
                       int n_samples,
                       int n_features,
                       const std::vector<int>& y,
                       const std::vector<int>& indices);

    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    int predict_one(const std::vector<double>& sample) const;

    void save_model(std::ostream& out) const;
    void load_model(std::istream& in);

private:
    std::unique_ptr<Node> root;
    int max_depth;
    int min_samples_split;
    int chunk_size;
    bool use_optimized_mode; // Apenas para saber qual fluxo seguir na recursão

    // Buffer de Reuso (Apenas para o método otimizado)
    std::vector<std::pair<double, int>> sort_buffer; 

    // Construtor Recursivo
    std::unique_ptr<Node> build_tree(const std::vector<std::vector<double>>* X_row,
                                     const std::vector<double>* X_flat,
                                     int n_total_samples,
                                     int n_features,
                                     const std::vector<int>& y,
                                     const std::vector<int>& indices,
                                     int depth);

    // --- IMPLEMENTAÇÃO 1: NAIVE ---
    // Acesso Row-Major, Alocação Local (Ineficiente)
    void find_best_split_naive(const std::vector<std::vector<double>>& X,
                               int n_features,
                               const std::vector<int>& y,
                               const std::vector<int>& indices,
                               int& best_feature,
                               double& best_threshold,
                               std::vector<int>& left_idx,
                               std::vector<int>& right_idx,
                               double parent_gini);

    // --- IMPLEMENTAÇÃO 2: OTIMIZADA ---
    // Acesso Column-Major, Reuso de Buffer, Chunks (Eficiente)
    void find_best_split_optimized(const std::vector<double>& X_flat,
                                   int n_total_samples,
                                   int n_features,
                                   const std::vector<int>& y,
                                   const std::vector<int>& indices,
                                   int& best_feature,
                                   double& best_threshold,
                                   std::vector<int>& left_idx,
                                   std::vector<int>& right_idx,
                                   double parent_gini);

    double calculate_gini(const std::vector<int>& labels) const;
    int majority_class(const std::vector<int>& labels) const;
    int predict_sample(const std::vector<double>& sample, const Node* node) const;

    void save_node(std::ostream& out, const Node* node) const;
    std::unique_ptr<Node> load_node(std::istream& in);
};

#endif // DECISION_TREE_H