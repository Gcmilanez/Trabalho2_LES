#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>
#include <cstddef>
#include <iosfwd>

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

    // ------------------------------
    // Construtor padrão
    // ------------------------------
    DecisionTree(int max_depth = 10,
                 int min_samples_split = 2,
                 int chunk_size = DEFAULT_CHUNK_SIZE);

    // ------------------------------
    // REMOVER cópia (não pode, por causa do unique_ptr)
    // ------------------------------
    DecisionTree(const DecisionTree&) = delete;
    DecisionTree& operator=(const DecisionTree&) = delete;

    // ------------------------------
    // ADICIONAR movimento (necessário para vector)
    // ------------------------------
    DecisionTree(DecisionTree&& other) noexcept;
    DecisionTree& operator=(DecisionTree&& other) noexcept;

    // ------------------------------
    // TREINO
    // ------------------------------
    void fit(const std::vector<std::vector<double>>& X,
             const std::vector<int>& y,
             bool use_chunks);

    // ------------------------------
    // PREDIÇÃO
    // ------------------------------
    int predict_one(const std::vector<double>& sample) const;
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;

    // ------------------------------
    // Contadores
    // ------------------------------
    

    void reset_access_counters() {
        cache_friendly_accesses = 0;
        random_accesses = 0;
    }

    // ------------------------------
    // GETTERS
    // ------------------------------
    int get_max_depth() const        { return max_depth; }
    int get_min_samples_split() const{ return min_samples_split; }
    int get_chunk_size() const       { return chunk_size; }
    bool get_use_chunks() const      { return use_chunk_processing; }

    void set_max_depth(int d)        { max_depth = d; }
    void set_min_samples_split(int m){ min_samples_split = m; }
    void set_chunk_size(int c)       { chunk_size = c; }
    void set_use_chunks(bool v)      { use_chunk_processing = v; }

    // ------------------------------
    // SERIALIZAÇÃO
    // ------------------------------
    void save_model(std::ostream& out) const;
    void load_model(std::istream& in);

private:
    // raiz da árvore
    std::unique_ptr<Node> root;

    // hiperparâmetros
    int max_depth;
    int min_samples_split;
    bool use_chunk_processing;
    int chunk_size;

    mutable std::size_t cache_friendly_accesses;
    mutable std::size_t random_accesses;

    // buffers internos
    std::vector<const std::vector<double>*> chunk_buffer;
    std::vector<int> chunk_labels_buffer;

    // ---- Funções auxiliares internas ----
    std::unique_ptr<Node> build_tree(const std::vector<std::vector<double>>& X,
                                     const std::vector<int>& y,
                                     const std::vector<int>& indices,
                                     int depth);

    double calculate_gini(const std::vector<int>& labels) const;
    int majority_class(const std::vector<int>& labels) const;

    void find_best_split_basic(const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y,
                               const std::vector<int>& indices,
                               int& best_feature,
                               double& best_threshold,
                               std::vector<int>& left_idx,
                               std::vector<int>& right_idx,
                               double parent_gini) const;

    void find_best_split_chunked(const std::vector<std::vector<double>>& X,
                                 const std::vector<int>& y,
                                 const std::vector<int>& indices,
                                 int& best_feature,
                                 double& best_threshold,
                                 std::vector<int>& left_idx,
                                 std::vector<int>& right_idx,
                                 double parent_gini);

    int predict_sample(const std::vector<double>& sample,
                       const Node* node) const;

    // serialização recursiva
    void save_node(std::ostream& out, const Node* node) const;
    std::unique_ptr<Node> load_node(std::istream& in);
};

#endif // DECISION_TREE_H
