#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>
#include <iostream>

struct Node {
    bool is_leaf = false;
    int predicted_class = -1;
    int feature_index = -1;
    double threshold = 0.0;
    std::unique_ptr<Node> left = nullptr;
    std::unique_ptr<Node> right = nullptr;
};

class DecisionTree {
public:
    // === CORREÇÃO DE COMPATIBILIDADE ===
    // Adicionamos esta constante para que a RandomForestOptimized pare de reclamar
    static const int DEFAULT_CHUNK_SIZE = 256;

    // Adicionamos o parametro chunk_size (que será ignorado) para compatibilidade
    DecisionTree(int max_depth = 10, int min_samples_split = 2, int chunk_size = DEFAULT_CHUNK_SIZE);

    // Construtores de movimento
    DecisionTree(DecisionTree&& other) noexcept;
    DecisionTree& operator=(DecisionTree&& other) noexcept;

    // Desabilita cópia
    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;

    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<int>& y, 
             bool use_chunks = false, 
             const std::vector<int>* bootstrap_indices = nullptr);
    
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    int predict_one(const std::vector<double>& sample) const;

    // Serialização
    void save_model(std::ostream& out) const;
    void load_model(std::istream& in);

private:
    std::unique_ptr<Node> root;
    int max_depth;
    int min_samples_split;
    int num_classes; 

    struct SampleEntry {
        double value;
        int label;
        int original_index;
    };

    std::unique_ptr<Node> build_tree(
        const std::vector<std::vector<double>>& X_col_major, 
        const std::vector<int>& y,
        const std::vector<int>& indices,
        int depth);

    void find_best_split(
        const std::vector<std::vector<double>>& X_col_major,
        const std::vector<int>& y,
        const std::vector<int>& indices,
        int& best_feature,
        double& best_threshold,
        std::vector<int>& left_idx,
        std::vector<int>& right_idx,
        double parent_gini);

    // Utilitários
    double calculate_gini_from_counts(const std::vector<int>& counts, int total) const;
    int predict_sample(const std::vector<double>& sample, const Node* node) const;
    
    // Serialização Helpers
    void save_node(std::ostream& out, const Node* node) const;
    std::unique_ptr<Node> load_node(std::istream& in);
};

#endif // DECISION_TREE_H