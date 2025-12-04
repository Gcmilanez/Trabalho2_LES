#include "DecisionTree.h"

#include <algorithm>
#include <numeric>
#include <random>
#include <cmath>
#include <iostream>
#include <fstream>
#include <unordered_map>

// ============================================================
// Construtor principal
// ============================================================
DecisionTree::DecisionTree(int max_depth,
                           int min_samples_split,
                           int chunk_size)
    : root(nullptr),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      use_chunk_processing(false),
      chunk_size(chunk_size),
      cache_friendly_accesses(0),
      random_accesses(0)
{
}

// ============================================================
// CONSTRUTOR DE MOVIMENTO
// ============================================================
DecisionTree::DecisionTree(DecisionTree&& other) noexcept
{
    root = std::move(other.root);

    max_depth = other.max_depth;
    min_samples_split = other.min_samples_split;
    use_chunk_processing = other.use_chunk_processing;
    chunk_size = other.chunk_size;

    cache_friendly_accesses = other.cache_friendly_accesses;
    random_accesses = other.random_accesses;

    chunk_buffer = std::move(other.chunk_buffer);
    chunk_labels_buffer = std::move(other.chunk_labels_buffer);
}

// ============================================================
// OPERADOR DE ATRIBUIÇÃO DE MOVIMENTO
// ============================================================
DecisionTree& DecisionTree::operator=(DecisionTree&& other) noexcept
{
    if (this != &other)
    {
        root = std::move(other.root);

        max_depth = other.max_depth;
        min_samples_split = other.min_samples_split;
        use_chunk_processing = other.use_chunk_processing;
        chunk_size = other.chunk_size;

        cache_friendly_accesses = other.cache_friendly_accesses;
        random_accesses = other.random_accesses;

        chunk_buffer = std::move(other.chunk_buffer);
        chunk_labels_buffer = std::move(other.chunk_labels_buffer);
    }
    return *this;
}

// ============================================================
// Treino da árvore
// ============================================================
void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       bool use_chunks)
{
    use_chunk_processing = use_chunks;
    reset_access_counters();

    std::vector<int> indices(X.size());
    std::iota(indices.begin(), indices.end(), 0);

    root = build_tree(X, y, indices, 0);
}

// ============================================================
// Predição de uma amostra
// ============================================================
int DecisionTree::predict_one(const std::vector<double>& sample) const
{
    return predict_sample(sample, root.get());
}

// ============================================================
// Predição de várias amostras
// ============================================================
std::vector<int> DecisionTree::predict(const std::vector<std::vector<double>>& X) const
{
    std::vector<int> out;
    out.reserve(X.size());

    for (const auto& s : X)
        out.push_back(predict_sample(s, root.get()));

    return out;
}

// ============================================================
// Predição recursiva
// ============================================================
int DecisionTree::predict_sample(const std::vector<double>& sample,
                                 const Node* node) const
{
    if (!node || node->is_leaf)
        return node ? node->predicted_class : -1;

    double value = sample[node->feature_index];

    if (value <= node->threshold) {
        ++cache_friendly_accesses;   // OK por ser mutable
        return predict_sample(sample, node->left.get());
    } else {
        ++random_accesses;           // OK por ser mutable
        return predict_sample(sample, node->right.get());
    }
}

// ============================================================
// Cálculo do gini
// ============================================================
double DecisionTree::calculate_gini(const std::vector<int>& labels) const
{
    if (labels.empty())
        return 0.0;

    std::unordered_map<int,int> count;
    for (int c : labels)
        count[c]++;

    double N = labels.size();
    double impurity = 1.0;

    for (auto& kv : count) {
        double p = kv.second / N;
        impurity -= p * p;
    }

    return impurity;
}

// ============================================================
// Classe majoritária
// ============================================================
int DecisionTree::majority_class(const std::vector<int>& labels) const
{
    std::unordered_map<int,int> freq;
    for (int v : labels)
        freq[v]++;

    int best_class = -1;
    int best_count = -1;

    for (auto& kv : freq) {
        if (kv.second > best_count) {
            best_class = kv.first;
            best_count = kv.second;
        }
    }
    return best_class;
}

// ============================================================
// Construção recursiva da árvore
// ============================================================
std::unique_ptr<Node> DecisionTree::build_tree(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int depth)
{
    // labels atuais
    std::vector<int> current_labels;
    current_labels.reserve(indices.size());
    for (int idx : indices)
        current_labels.push_back(y[idx]);

    // condições de parada
    double gini_parent = calculate_gini(current_labels);
    if (depth >= max_depth ||
        indices.size() < (size_t)min_samples_split ||
        gini_parent == 0.0)
    {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority_class(current_labels);
        return leaf;
    }

    int best_feature = -1;
    double best_threshold = 0.0;
    std::vector<int> left_idx, right_idx;

    if (!use_chunk_processing)
        find_best_split_basic(X, y, indices,
                              best_feature, best_threshold,
                              left_idx, right_idx, gini_parent);
    else
        find_best_split_chunked(X, y, indices,
                                best_feature, best_threshold,
                                left_idx, right_idx, gini_parent);

    if (best_feature == -1 || left_idx.empty() || right_idx.empty()) {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority_class(current_labels);
        return leaf;
    }

    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;

    node->left  = build_tree(X, y, left_idx,  depth + 1);
    node->right = build_tree(X, y, right_idx, depth + 1);

    return node;
}

// ============================================================
// Split básico
// ============================================================
void DecisionTree::find_best_split_basic(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int& best_feature,
    double& best_threshold,
    std::vector<int>& left_idx,
    std::vector<int>& right_idx,
    double parent_gini) const
{
    const int n_features = X[0].size();
    double best_gain = 1e-12;

    for (int f = 0; f < n_features; f++) {
        for (int idx : indices) {

            double thr = X[idx][f];
            std::vector<int> l, r;

            for (int id : indices) {
                if (X[id][f] <= thr)
                    l.push_back(id);
                else
                    r.push_back(id);
            }

            if (l.empty() || r.empty())
                continue;

            auto build_vec = [&](const std::vector<int>& ids) {
                std::vector<int> temp;
                temp.reserve(ids.size());
                for (int i : ids) temp.push_back(y[i]);
                return temp;
            };

            double g_l = calculate_gini(build_vec(l));
            double g_r = calculate_gini(build_vec(r));

            double g = parent_gini -
                (l.size() / (double)indices.size()) * g_l -
                (r.size() / (double)indices.size()) * g_r;

            if (g > best_gain) {
                best_gain = g;
                best_feature = f;
                best_threshold = thr;
                left_idx = l;
                right_idx = r;
            }
        }
    }
}

// ============================================================
// Split com chunks (versão simplificada)
// ============================================================
void DecisionTree::find_best_split_chunked(
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int& best_feature,
    double& best_threshold,
    std::vector<int>& left_idx,
    std::vector<int>& right_idx,
    double parent_gini)
{
    const int n_features = X[0].size();
    double best_gain = 1e-12;

    for (int fchunk = 0; fchunk < n_features; fchunk += chunk_size) {
        int fend = std::min(fchunk + chunk_size, n_features);

        for (int f = fchunk; f < fend; f++) {

            // threshold = média dos valores
            double avg = 0.0;
            for (int idx : indices)
                avg += X[idx][f];
            avg /= indices.size();

            std::vector<int> l, r;
            for (int id : indices) {
                if (X[id][f] <= avg)
                    l.push_back(id);
                else
                    r.push_back(id);
            }

            if (l.empty() || r.empty())
                continue;

            auto build_vec = [&](const std::vector<int>& ids) {
                std::vector<int> temp;
                temp.reserve(ids.size());
                for (int i : ids) temp.push_back(y[i]);
                return temp;
            };

            double g_l = calculate_gini(build_vec(l));
            double g_r = calculate_gini(build_vec(r));

            double g = parent_gini -
                (l.size() / (double)indices.size()) * g_l -
                (r.size() / (double)indices.size()) * g_r;

            if (g > best_gain) {
                best_gain = g;
                best_feature = f;
                best_threshold = avg;
                left_idx = l;
                right_idx = r;
            }
        }
    }
}

// ============================================================
// SERIALIZAÇÃO
// ============================================================
void DecisionTree::save_model(std::ostream& out) const
{
    out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    out.write(reinterpret_cast<const char*>(&min_samples_split), sizeof(min_samples_split));
    out.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));
    out.write(reinterpret_cast<const char*>(&use_chunk_processing), sizeof(use_chunk_processing));

    save_node(out, root.get());
}

void DecisionTree::save_node(std::ostream& out, const Node* node) const
{
    bool exists = (node != nullptr);
    out.write(reinterpret_cast<const char*>(&exists), sizeof(bool));
    if (!exists) return;

    out.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(int));
    out.write(reinterpret_cast<const char*>(&node->threshold), sizeof(double));
    out.write(reinterpret_cast<const char*>(&node->predicted_class), sizeof(int));

    save_node(out, node->left.get());
    save_node(out, node->right.get());
}

// ============================================================
// DESERIALIZAÇÃO
// ============================================================
void DecisionTree::load_model(std::istream& in)
{
    in.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
    in.read(reinterpret_cast<char*>(&min_samples_split), sizeof(min_samples_split));
    in.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));
    in.read(reinterpret_cast<char*>(&use_chunk_processing), sizeof(use_chunk_processing));

    root = load_node(in);
}

std::unique_ptr<Node> DecisionTree::load_node(std::istream& in)
{
    bool exists = false;
    in.read(reinterpret_cast<char*>(&exists), sizeof(bool));
    if (!exists) return nullptr;

    auto node = std::make_unique<Node>();

    in.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(bool));
    in.read(reinterpret_cast<char*>(&node->feature_index), sizeof(int));
    in.read(reinterpret_cast<char*>(&node->threshold), sizeof(double));
    in.read(reinterpret_cast<char*>(&node->predicted_class), sizeof(int));

    node->left  = load_node(in);
    node->right = load_node(in);

    return node;
}
