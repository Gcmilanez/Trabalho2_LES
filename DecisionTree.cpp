#include "DecisionTree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <cstring>

DecisionTree::DecisionTree(int max_depth, int min_samples_split, int chunk_size)
    : root(nullptr), max_depth(max_depth), min_samples_split(min_samples_split), chunk_size(chunk_size), use_optimized_mode(false) {}

// ============================================================
// ENTRY POINTS
// ============================================================
void DecisionTree::fit_baseline(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y,
                                const std::vector<int>& indices)
{
    this->use_optimized_mode = false;
    if (X.empty()) return;
    int n_samples = X.size();
    int n_features = X[0].size();
    root = build_tree(&X, nullptr, n_samples, n_features, y, indices, 0);
}

void DecisionTree::fit_optimized(const std::vector<double>& X_flat,
                                 int n_samples,
                                 int n_features,
                                 const std::vector<int>& y,
                                 const std::vector<int>& indices)
{
    this->use_optimized_mode = true;
    sort_buffer.reserve(n_samples); // Pre-aloca uma vez
    root = build_tree(nullptr, &X_flat, n_samples, n_features, y, indices, 0);
}

// ============================================================
// BUILD TREE (RECURSÃO)
// ============================================================
std::unique_ptr<Node> DecisionTree::build_tree(
    const std::vector<std::vector<double>>* X_row,
    const std::vector<double>* X_flat,
    int n_total_samples,
    int n_features,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int depth)
{
    std::vector<int> current_labels;
    current_labels.reserve(indices.size());
    for (int idx : indices) current_labels.push_back(y[idx]);

    double gini = calculate_gini(current_labels);

    if (depth >= max_depth || indices.size() < (size_t)min_samples_split || gini == 0.0) {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority_class(current_labels);
        return leaf;
    }

    int best_feature = -1;
    double best_threshold = 0.0;
    std::vector<int> left_idx, right_idx;

    // SEPARAÇÃO LIMPA DE LÓGICA
    if (use_optimized_mode) {
        find_best_split_optimized(*X_flat, n_total_samples, n_features, y, indices,
                                  best_feature, best_threshold, left_idx, right_idx, gini);
    } else {
        find_best_split_naive(*X_row, n_features, y, indices,
                              best_feature, best_threshold, left_idx, right_idx, gini);
    }

    if (best_feature == -1) {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority_class(current_labels);
        return leaf;
    }

    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    
    node->left = build_tree(X_row, X_flat, n_total_samples, n_features, y, left_idx, depth + 1);
    node->right = build_tree(X_row, X_flat, n_total_samples, n_features, y, right_idx, depth + 1);

    return node;
}

// ============================================================
// 1. NAIVE SPLIT (Exatamente como solicitado)
// ============================================================
void DecisionTree::find_best_split_naive(
    const std::vector<std::vector<double>>& X,
    int n_features,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int& best_feature,
    double& best_threshold,
    std::vector<int>& left_idx,
    std::vector<int>& right_idx,
    double parent_gini)
{
    const size_t n_node_samples = indices.size();
    double best_gain = -1.0;

    // [Ineficiência 1] Alocação local a cada chamada
    std::vector<std::pair<double, int>> values;
    values.reserve(n_node_samples);

    int max_label = 0; for(int idx : indices) if(y[idx] > max_label) max_label = y[idx];
    int num_classes = max_label + 1;
    std::vector<int> right_counts(num_classes);
    std::vector<int> left_counts(num_classes);

    for (int f = 0; f < n_features; f++) {
        // [Ineficiência 2] GATHER Row-Major (Pulo de memória)
        values.clear();
        for (int idx : indices) {
            values.push_back({ X[idx][f], idx });
        }

        std::sort(values.begin(), values.end());

        std::fill(left_counts.begin(), left_counts.end(), 0);
        std::fill(right_counts.begin(), right_counts.end(), 0);
        for (int idx : indices) right_counts[y[idx]]++;

        double sum_sq_right = 0.0;
        for (int c : right_counts) if (c > 0) sum_sq_right += (double)c * c;
        double sum_sq_left = 0.0;
        double size_left = 0;
        double size_right = (double)n_node_samples;

        for (size_t i = 0; i < n_node_samples - 1; i++) {
            int idx = values[i].second;
            int label = y[idx];

            int c_r = right_counts[label];
            sum_sq_right -= (double)c_r * c_r;
            right_counts[label]--;
            sum_sq_right += (double)right_counts[label] * right_counts[label];

            int c_l = left_counts[label];
            sum_sq_left -= (double)c_l * c_l;
            left_counts[label]++;
            sum_sq_left += (double)left_counts[label] * left_counts[label];
            
            size_left++; size_right--;

            if (values[i].first == values[i+1].first) continue;

            double gini_left = 1.0 - (sum_sq_left / (size_left * size_left));
            double gini_right = 1.0 - (sum_sq_right / (size_right * size_right));
            double gain = parent_gini - (size_left/n_node_samples * gini_left) - (size_right/n_node_samples * gini_right);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = (values[i].first + values[i+1].first) / 2.0;
            }
        }
    }

    if (best_gain > 0.0) {
        left_idx.reserve(n_node_samples); 
        right_idx.reserve(n_node_samples);
        for (int idx : indices) {
            if (X[idx][best_feature] <= best_threshold) left_idx.push_back(idx);
            else right_idx.push_back(idx);
        }
    }
}

// ============================================================
// 2. OPTIMIZED SPLIT (Exato + Cache Friendly)
// ============================================================
void DecisionTree::find_best_split_optimized(
    const std::vector<double>& X_flat,
    int n_total_samples,
    int n_features,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int& best_feature,
    double& best_threshold,
    std::vector<int>& left_idx,
    std::vector<int>& right_idx,
    double parent_gini)
{
    double best_gain = -1.0;
    size_t n_node_samples = indices.size();

    int max_label = 0; for(int idx : indices) if (y[idx] > max_label) max_label = y[idx];
    int num_classes = max_label + 1;

    // [Otimização] Hoisting de vetores
    std::vector<int> right_counts(num_classes);
    std::vector<int> left_counts(num_classes);

    for (int f = 0; f < n_features; f++) {
        
        // [Otimização] Reuso do sort_buffer (Zero Alloc)
        sort_buffer.clear(); 
        const double* feature_ptr = &X_flat[f * n_total_samples];

        // [Otimização] GATHER COM CHUNKS (Cache L1)
        for (size_t chunk_start = 0; chunk_start < n_node_samples; chunk_start += chunk_size) {
            size_t chunk_end = std::min(chunk_start + (size_t)chunk_size, n_node_samples);
            for (size_t i = chunk_start; i < chunk_end; i++) {
                int idx = indices[i];
                sort_buffer.push_back({ feature_ptr[idx], idx });
            }
        }
        
        // [Otimização] A partir daqui, igual ao Naive (Sort + Linear Scan)
        std::sort(sort_buffer.begin(), sort_buffer.end());

        std::memset(left_counts.data(), 0, num_classes * sizeof(int));
        std::memset(right_counts.data(), 0, num_classes * sizeof(int));
        for (int idx : indices) right_counts[y[idx]]++;

        double sum_sq_right = 0.0;
        for (int c : right_counts) if (c > 0) sum_sq_right += (double)c * c;
        double sum_sq_left = 0.0;
        double size_left = 0;
        double size_right = (double)n_node_samples;

        for (size_t i = 0; i < n_node_samples - 1; i++) {
            int idx = sort_buffer[i].second;
            int label = y[idx];

            int c_r = right_counts[label];
            sum_sq_right -= (double)c_r * c_r;
            right_counts[label]--;
            sum_sq_right += (double)right_counts[label] * right_counts[label];

            int c_l = left_counts[label];
            sum_sq_left -= (double)c_l * c_l;
            left_counts[label]++;
            sum_sq_left += (double)left_counts[label] * left_counts[label];
            
            size_left++; size_right--;

            if (sort_buffer[i].first == sort_buffer[i+1].first) continue;

            double gini_left = 1.0 - (sum_sq_left / (size_left * size_left));
            double gini_right = 1.0 - (sum_sq_right / (size_right * size_right));
            double gain = parent_gini - (size_left/n_node_samples * gini_left) - (size_right/n_node_samples * gini_right);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = (sort_buffer[i].first + sort_buffer[i+1].first) / 2.0;
            }
        }
    }

    if (best_gain > 0.0) {
        left_idx.reserve(n_node_samples); right_idx.reserve(n_node_samples);
        const double* feature_ptr = &X_flat[best_feature * n_total_samples];
        for (int idx : indices) {
            // Usa acesso rápido para reconstrução também
            if (feature_ptr[idx] <= best_threshold) left_idx.push_back(idx);
            else right_idx.push_back(idx);
        }
    }
}

// Helpers Padrão (Sem alterações)
double DecisionTree::calculate_gini(const std::vector<int>& labels) const {
    if (labels.empty()) return 0.0;
    std::unordered_map<int, int> counts;
    for (int l : labels) counts[l]++;
    double imp = 1.0; double n = (double)labels.size();
    for (auto& kv : counts) imp -= pow(kv.second / n, 2);
    return imp;
}

int DecisionTree::majority_class(const std::vector<int>& labels) const {
    if (labels.empty()) return -1;
    std::unordered_map<int, int> counts;
    int best_c = -1, best_cnt = -1;
    for (int l : labels) { counts[l]++; if (counts[l] > best_cnt) { best_cnt = counts[l]; best_c = l; } }
    return best_c;
}

int DecisionTree::predict_one(const std::vector<double>& sample) const {
    return predict_sample(sample, root.get());
}

std::vector<int> DecisionTree::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> res; res.reserve(X.size());
    for(const auto& s : X) res.push_back(predict_one(s));
    return res;
}

int DecisionTree::predict_sample(const std::vector<double>& sample, const Node* node) const {
    if (!node || node->is_leaf) return node ? node->predicted_class : -1;
    if (sample[node->feature_index] <= node->threshold) return predict_sample(sample, node->left.get());
    else return predict_sample(sample, node->right.get());
}

void DecisionTree::save_model(std::ostream& out) const {
    out.write((char*)&max_depth, sizeof(int));
    out.write((char*)&min_samples_split, sizeof(int));
    out.write((char*)&chunk_size, sizeof(int));
    save_node(out, root.get());
}

void DecisionTree::save_node(std::ostream& out, const Node* node) const {
    bool exists = (node != nullptr);
    out.write((char*)&exists, sizeof(bool));
    if(!exists) return;
    out.write((char*)&node->is_leaf, sizeof(bool));
    out.write((char*)&node->feature_index, sizeof(int));
    out.write((char*)&node->threshold, sizeof(double));
    out.write((char*)&node->predicted_class, sizeof(int));
    save_node(out, node->left.get());
    save_node(out, node->right.get());
}

void DecisionTree::load_model(std::istream& in) {
    in.read((char*)&max_depth, sizeof(int));
    in.read((char*)&min_samples_split, sizeof(int));
    in.read((char*)&chunk_size, sizeof(int));
    root = load_node(in);
}

std::unique_ptr<Node> DecisionTree::load_node(std::istream& in) {
    bool ex; in.read((char*)&ex, sizeof(bool));
    if(!ex) return nullptr;
    auto node = std::make_unique<Node>();
    in.read((char*)&node->is_leaf, sizeof(bool));
    in.read((char*)&node->feature_index, sizeof(int));
    in.read((char*)&node->threshold, sizeof(double));
    in.read((char*)&node->predicted_class, sizeof(int));
    node->left = load_node(in);
    node->right = load_node(in);
    return node;
}