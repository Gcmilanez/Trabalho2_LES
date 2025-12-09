#include "DecisionTree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <cstring>
#include <limits>
#include <unordered_map>

// ============================================================
// CONSTRUTOR
// ============================================================
DecisionTree::DecisionTree(int max_depth, int min_samples_split, int chunk_size)
    : root(nullptr),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      chunk_size(chunk_size),
      use_optimized_mode(false)
{
}

// ============================================================
// FIT
// ============================================================
void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       bool use_optimized,
                       const std::vector<int>* bootstrap_indices)
{
    this->use_optimized_mode = use_optimized;
    int n_samples = X.size();
    if (n_samples == 0) return;
    int n_features = X[0].size();
    
    std::vector<int> indices;
    if (bootstrap_indices) indices = *bootstrap_indices;
    else {
        indices.resize(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
    }

    if (!use_optimized) {
        // BASELINE (Exato)
        root = build_tree(&X, nullptr, n_samples, n_features, y, indices, 0);
    } 
    else {
        // OTIMIZADO (Histograma)
        std::vector<uint8_t> X_binned;
        bin_thresholds.clear();
        
        // 1. Discretização (Gera bins variados por feature)
        discretize_features(X, X_binned, bin_thresholds);
        
        root = build_tree(nullptr, &X_binned, n_samples, n_features, y, indices, 0);
    }
}

// ============================================================
// DISCRETIZAÇÃO (QUANTILE)
// ============================================================
void DecisionTree::discretize_features(const std::vector<std::vector<double>>& X,
                                       std::vector<uint8_t>& X_binned,
                                       std::vector<std::vector<double>>& thresholds)
{
    int n_samples = X.size();
    int n_features = X[0].size();
    
    X_binned.resize(n_samples * n_features);
    thresholds.resize(n_features);

    const int max_bins = 255; 
    std::vector<double> feature_values(n_samples);

    for (int f = 0; f < n_features; f++) {
        for(int i=0; i<n_samples; ++i) feature_values[i] = X[i][f];
        std::sort(feature_values.begin(), feature_values.end());
        auto last = std::unique(feature_values.begin(), feature_values.end());
        int unique_count = std::distance(feature_values.begin(), last);
        
        thresholds[f].clear();
        if (unique_count <= max_bins) {
            for(int i=0; i<unique_count; ++i) thresholds[f].push_back(feature_values[i]);
        } else {
            double step = (double)unique_count / max_bins;
            for(int b=1; b < max_bins; ++b) {
                int idx = (int)(b * step);
                if (idx < unique_count) thresholds[f].push_back(feature_values[idx]);
            }
            thresholds[f].push_back(feature_values[unique_count-1]);
        }

        const auto& thr = thresholds[f];
        for(int i=0; i<n_samples; ++i) {
            double val = X[i][f];
            auto it = std::lower_bound(thr.begin(), thr.end(), val);
            int bin = std::distance(thr.begin(), it);
            if (bin >= (int)thr.size()) bin = (int)thr.size() - 1; 
            X_binned[f * n_samples + i] = (uint8_t)bin;
        }
    }
}

// ============================================================
// BUILD TREE
// ============================================================
std::unique_ptr<Node> DecisionTree::build_tree(
    const std::vector<std::vector<double>>* X_naive,
    const std::vector<uint8_t>* X_binned,
    int n_total_samples,
    int n_features,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int depth)
{
    int first_label = y[indices[0]];
    bool pure = true;
    for(size_t i=1; i<indices.size(); ++i) {
        if(y[indices[i]] != first_label) { pure = false; break; }
    }

    if (depth >= max_depth || indices.size() < (size_t)min_samples_split || pure) {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        int best_c = -1, max_cnt = -1;
        int max_lbl = 0; for(int idx : indices) if(y[idx]>max_lbl) max_lbl=y[idx];
        std::vector<int> counts(max_lbl+1, 0);
        for(int idx : indices) {
            counts[y[idx]]++;
            if(counts[y[idx]] > max_cnt) { max_cnt = counts[y[idx]]; best_c = y[idx]; }
        }
        leaf->predicted_class = best_c;
        return leaf;
    }

    double parent_gini = 1.0; 
    int best_feature = -1;
    double best_threshold = 0.0;
    std::vector<int> left_idx, right_idx;

    if (!use_optimized_mode) {
        find_best_split_naive(*X_naive, y, indices, 
                              best_feature, best_threshold, left_idx, right_idx, parent_gini);
    } else {
        find_best_split_histogram(*X_binned, n_total_samples, n_features, y, indices,
                                  best_feature, best_threshold, left_idx, right_idx, parent_gini);
    }

    if (best_feature == -1) {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority_class(std::vector<int>()); // Fallback dummy
        // Recalcula maioria localmente para evitar overhead de vector copy
        int best_c = -1, max_cnt = -1;
        int max_l = 0; for(int idx : indices) if(y[idx]>max_l) max_l=y[idx];
        std::vector<int> c(max_l+1, 0);
        for(int idx : indices) {
            c[y[idx]]++;
            if(c[y[idx]]>max_cnt){ max_cnt=c[y[idx]]; best_c=y[idx];}
        }
        leaf->predicted_class = best_c;
        return leaf;
    }

    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    
    node->left = build_tree(X_naive, X_binned, n_total_samples, n_features, y, left_idx, depth + 1);
    node->right = build_tree(X_naive, X_binned, n_total_samples, n_features, y, right_idx, depth + 1);

    return node;
}

// ============================================================
// NAIVE SPLIT (BASELINE)
// ============================================================
void DecisionTree::find_best_split_naive(
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
    const size_t n_node_samples = indices.size();
    double best_gain = -1.0;

    // Otimização: Aloca buffer FORA do loop de features para reutilizar capacidade
    std::vector<std::pair<double, int>> values;
    values.reserve(n_node_samples);

    // Setup de classes (igual ao Sklearn, usa arrays)
    int max_label = 0;
    for(int idx : indices) if(y[idx] > max_label) max_label = y[idx];
    int num_classes = max_label + 1;

    std::vector<int> right_counts(num_classes);
    std::vector<int> left_counts(num_classes);

    for (int f = 0; f < n_features; f++) {
        
        // 1. GATHER (Padrão C++)
        // Acesso X[idx][f] é o padrão para vector<vector>.
        // Não é "ineficiência proposital", é apenas como a estrutura funciona.
        values.clear();
        for (int idx : indices) {
            values.push_back({ X[idx][f], idx });
        }

        // 2. SORT (Algoritmo Eficiente)
        std::sort(values.begin(), values.end());

        // 3. SCAN LINEAR INCREMENTAL (Exatamente como o Sklearn faz)
        std::fill(left_counts.begin(), left_counts.end(), 0);
        std::fill(right_counts.begin(), right_counts.end(), 0);

        // Popula direita
        for (int idx : indices) right_counts[y[idx]]++;

        // Soma dos Quadrados (Matemática Rápida)
        double sum_sq_right = 0.0;
        for (int c : right_counts) if (c > 0) sum_sq_right += (double)c * c;
        
        double sum_sq_left = 0.0;
        double size_left = 0;
        double size_right = (double)n_node_samples;

        // Varredura única O(N)
        for (size_t i = 0; i < n_node_samples - 1; i++) {
            
            int idx = values[i].second;
            int label = y[idx];

            // Atualização O(1)
            int count_r = right_counts[label];
            sum_sq_right -= (double)count_r * count_r;
            right_counts[label]--;
            sum_sq_right += (double)right_counts[label] * right_counts[label];

            int count_l = left_counts[label];
            sum_sq_left -= (double)count_l * count_l;
            left_counts[label]++;
            sum_sq_left += (double)left_counts[label] * left_counts[label];
            
            size_left++; size_right--;

            // Pula duplicatas
            if (values[i].first == values[i+1].first) continue;

            // Gini Gain
            double gini_left = 1.0 - (sum_sq_left / (size_left * size_left));
            double gini_right = 1.0 - (sum_sq_right / (size_right * size_right));

            double gain = parent_gini - 
                          (size_left / n_node_samples * gini_left) - 
                          (size_right / n_node_samples * gini_right);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = (values[i].first + values[i+1].first) / 2.0;
            }
        }
    }

    // Reconstrói índices
    if (best_gain > 0.0) {
        left_idx.reserve(n_node_samples); 
        right_idx.reserve(n_node_samples);
        for (int idx : indices) {
            if (X[idx][best_feature] <= best_threshold)
                left_idx.push_back(idx);
            else
                right_idx.push_back(idx);
        }
    }
}

// ============================================================
// HISTOGRAM SPLIT (CORRIGIDO)
// ============================================================
void DecisionTree::find_best_split_histogram(
    const std::vector<uint8_t>& X_binned,
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

    std::vector<int> parent_counts(num_classes, 0);
    for(int idx : indices) parent_counts[y[idx]]++;
    double sum_sq_parent = 0.0;
    for(int c : parent_counts) if(c>0) sum_sq_parent += (double)c*c;
    parent_gini = 1.0 - (sum_sq_parent / (double)(n_node_samples*n_node_samples));

    // Vetores reutilizados
    const int max_possible_bins = 256;
    std::vector<int> histogram(max_possible_bins * num_classes, 0);
    std::vector<int> left_counts(num_classes);
    std::vector<int> right_counts(num_classes);

    for (int f = 0; f < n_features; f++) {
        
        // [FIX] Pega o número REAL de bins desta feature
        int n_bins_feature = bin_thresholds[f].size();
        if (n_bins_feature <= 1) continue; // Sem split possível

        // 1. Constroi Histograma (Zera apenas o necessário)
        std::fill(histogram.begin(), histogram.begin() + (n_bins_feature * num_classes), 0);
        
        const uint8_t* feature_ptr = &X_binned[f * n_total_samples];
        for(int idx : indices) {
            uint8_t bin = feature_ptr[idx];
            // Segurança extra: se bin >= n_bins_feature, ignora ou clamp (não deve ocorrer com discretize correto)
            if (bin < n_bins_feature) {
                histogram[bin * num_classes + y[idx]]++;
            }
        }

        // 2. Scan Linear nos Bins Reais
        std::fill(left_counts.begin(), left_counts.end(), 0);
        right_counts = parent_counts; 

        double sum_sq_right = sum_sq_parent;
        double sum_sq_left = 0.0;
        double size_left = 0;
        double size_right = (double)n_node_samples;

        // [FIX] Loop vai apenas até n_bins_feature - 1
        for (int b = 0; b < n_bins_feature - 1; ++b) {
            int bin_total = 0;
            for (int c = 0; c < num_classes; ++c) {
                int count = histogram[b * num_classes + c];
                if (count == 0) continue;
                bin_total += count;
                
                // Update
                int r_old = right_counts[c];
                sum_sq_right -= (double)r_old*r_old;
                right_counts[c] -= count;
                sum_sq_right += (double)right_counts[c]*right_counts[c];

                int l_old = left_counts[c];
                sum_sq_left -= (double)l_old*l_old;
                left_counts[c] += count;
                sum_sq_left += (double)left_counts[c]*left_counts[c];
            }

            size_left += bin_total;
            size_right -= bin_total;

            if (size_left == 0 || size_right == 0) continue;

            double gini_left = 1.0 - (sum_sq_left / (size_left * size_left));
            double gini_right = 1.0 - (sum_sq_right / (size_right * size_right));
            double gain = parent_gini - (size_left/n_node_samples * gini_left) - (size_right/n_node_samples * gini_right);

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                // [FIX] Acesso seguro ao threshold
                best_threshold = bin_thresholds[f][b];
            }
        }
    }

    if (best_gain > 0.0) {
        left_idx.reserve(n_node_samples); 
        right_idx.reserve(n_node_samples);
        const uint8_t* feature_ptr = &X_binned[best_feature * n_total_samples];
        
        // [FIX] Recuperação robusta do Split Bin
        int split_bin = -1;
        const auto& bins = bin_thresholds[best_feature];
        for(size_t b=0; b<bins.size(); ++b) {
            if (std::abs(bins[b] - best_threshold) < 1e-9) {
                split_bin = b; break;
            }
        }

        for (int idx : indices) {
            // Nota: Se split_bin for -1 (erro numérico), tudo vai pra direita, evitando crash
            if ((int)feature_ptr[idx] <= split_bin) left_idx.push_back(idx);
            else right_idx.push_back(idx);
        }
    }
}

// ============================================================
// HELPERS E SERIALIZAÇÃO (INALTERADOS)
// ============================================================
double DecisionTree::calculate_gini(const std::vector<int>& labels) const {
    if (labels.empty()) return 0.0;
    int max_l=0; for(int l:labels) if(l>max_l) max_l=l;
    std::vector<int> c(max_l+1, 0);
    for(int l:labels) c[l]++;
    double imp = 1.0; double n = (double)labels.size();
    for(int cnt : c) if(cnt>0) imp -= (double)(cnt*cnt)/(n*n);
    return imp;
}

int DecisionTree::majority_class(const std::vector<int>& labels) const {
    if(labels.empty()) return -1;
    int max_l=0; for(int l:labels) if(l>max_l) max_l=l;
    std::vector<int> counts(max_l+1, 0);
    int best=-1, max_c=-1;
    for(int l:labels) {
        counts[l]++;
        if(counts[l]>max_c) { max_c=counts[l]; best=l; }
    }
    return best;
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
    out.write((char*)&use_optimized_mode, sizeof(bool));
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
    in.read((char*)&use_optimized_mode, sizeof(bool));
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