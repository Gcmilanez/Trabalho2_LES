#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>

struct Node {
    bool is_leaf = false;
    int feature_index = -1;
    double threshold = 0.0;
    int predicted_class = -1;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

class DecisionTree {
private:
    int max_depth;
    int min_samples_split;
    std::unique_ptr<Node> root;
    
    // Flag para usar processamento em chunks
    bool use_chunk_processing;
    static const int CHUNK_SIZE = 100; // Tamanho do bloco para cache locality
    
    // Buffers pré-alocados para versão com chunks
    std::vector<double> chunk_buffer;
    std::vector<int> chunk_labels_buffer;
    
    // Métodos auxiliares
    double calculate_gini(const std::vector<int>& labels) const;
    
    struct SplitResult {
        double threshold;
        double gain;
        std::vector<int> left_indices;
        std::vector<int> right_indices;
    };
    
    // Versão BÁSICA - acesso aleatório
    SplitResult find_best_split_basic(const std::vector<int>& indices,
                                      int feature_index,
                                      const std::vector<std::vector<double>>& X,
                                      const std::vector<int>& y);
    
    // Versão OTIMIZADA - processamento em chunks
    SplitResult find_best_split_chunked(const std::vector<int>& indices,
                                        int feature_index,
                                        const std::vector<std::vector<double>>& X,
                                        const std::vector<int>& y);
    
    std::unique_ptr<Node> build_tree(const std::vector<int>& indices,
                                     const std::vector<std::vector<double>>& X,
                                     const std::vector<int>& y,
                                     int depth);
    
    int predict_sample(const std::vector<double>& sample, const Node* node) const;
    
public:
    DecisionTree(int max_depth = 5, int min_samples_split = 2);
    
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<int>& y,
             bool use_chunks = false);
    
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
};

#endif
