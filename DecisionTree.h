#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>
#include <utility>
#include <iostream>
#include <random>
#include <algorithm>
#include <numeric>

// [MANTIDO IGUAL] Struct alinhada para Cache
struct alignas(32) FlatNode {
    double threshold;       
    int feature_index;      
    int right_child_offset; 
    int predicted_class;    
    char padding[12];       
};

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
    static constexpr int DEFAULT_CHUNK_SIZE = 32;

    DecisionTree(int max_depth = 10, int min_samples_split = 2, int chunk_size = DEFAULT_CHUNK_SIZE);
    
    // Construtor move e delete copy [MANTIDO IGUAL]
    DecisionTree() : DecisionTree(10, 2, DEFAULT_CHUNK_SIZE) {}
    DecisionTree(DecisionTree&&) noexcept = default;
    DecisionTree& operator=(DecisionTree&&) noexcept = default;
    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;

    // Treino [MANTIDO IGUAL]
    void fit_baseline(const std::vector<double>& X_flat, int n_samples, int n_features, const std::vector<int>& y, const std::vector<int>& indices);
    void fit_optimized(const std::vector<double>& X_flat, int n_samples, int n_features, const std::vector<int>& y, const std::vector<int>& indices);

    // Predição
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    int predict_one(const std::vector<double>& sample) const;
    
    // [NOVO] Método de ultra-baixa latência que aceita ponteiro cru
    int predict_sample_flat_ptr(const double* sample) const;

    // Serialização [MANTIDO IGUAL]
    void save_model(std::ostream& out) const;
    void load_model(std::istream& in);

private:
    std::unique_ptr<Node> root;
    std::vector<FlatNode> flat_tree; 
    bool is_flat_built; 
    
    int max_depth;
    int min_samples_split;
    int chunk_size;
    bool use_optimized_mode; 
    std::vector<std::pair<double, int>> sort_buffer; 

    // Métodos privados auxiliares [MANTIDOS IGUAIS]
    std::unique_ptr<Node> build_tree(const std::vector<double>* X_flat, int n, int f, const std::vector<int>& y, const std::vector<int>& idx, int d);
    void find_best_split_naive(const std::vector<double>& X_flat, int n, int f, const std::vector<int>& y, const std::vector<int>& idx, int& bf, double& bt, std::vector<int>& l, std::vector<int>& r, double pg);
    void find_best_split_optimized(const std::vector<double>& X_flat, int n, int f, const std::vector<int>& y, const std::vector<int>& idx, int& bf, double& bt, std::vector<int>& l, std::vector<int>& r, double pg);

    double calculate_gini(const std::vector<int>& labels) const;
    int majority_class(const std::vector<int>& labels) const;

    int predict_sample_ptr(const std::vector<double>& sample, const Node* node) const;
    int predict_sample_flat(const std::vector<double>& sample) const;

    void flatten_tree(); 
    int fill_flat_tree(const Node* node); 

    void save_node(std::ostream& out, const Node* node) const;
    std::unique_ptr<Node> load_node(std::istream& in);
};

#endif // DECISION_TREE_H