#include "DecisionTree.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <limits>

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : max_depth(max_depth), min_samples_split(min_samples_split),
      use_chunk_processing(false) {
    // Pré-alocação de buffers para versão com chunks
    chunk_buffer.reserve(CHUNK_SIZE);
    chunk_labels_buffer.reserve(CHUNK_SIZE);
}

double DecisionTree::calculate_gini(const std::vector<int>& labels) const {
    if (labels.empty()) return 0.0;
    
    std::map<int, int> class_counts;
    for (int label : labels) {
        class_counts[label]++;
    }
    
    double gini = 1.0;
    double n = static_cast<double>(labels.size());
    
    for (const auto& pair : class_counts) {
        double prob = pair.second / n;
        gini -= prob * prob;
    }
    
    return gini;
}

DecisionTree::SplitResult DecisionTree::find_best_split_basic(
    const std::vector<int>& indices,
    int feature_index,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    
    // VERSÃO BÁSICA - Acesso aleatório aos dados (CACHE MISS frequente)
    std::vector<double> feature_values;
    feature_values.reserve(indices.size());
    
    // Acesso não sequencial - pior performance de cache
    for (int idx : indices) {
        feature_values.push_back(X[idx][feature_index]);
    }
    
    std::vector<double> unique_values = feature_values;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), 
                       unique_values.end());
    
    double best_gain = -1.0;
    double best_threshold = 0.0;
    std::vector<int> best_left, best_right;
    
    // Calcular gini do pai
    std::vector<int> current_labels;
    current_labels.reserve(indices.size());
    for (int idx : indices) {
        current_labels.push_back(y[idx]);
    }
    double parent_gini = calculate_gini(current_labels);
    
    // Testar cada threshold
    for (size_t i = 0; i < unique_values.size() - 1; ++i) {
        double threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;
        
        std::vector<int> left_labels, right_labels;
        std::vector<int> left_idx, right_idx;
        
        // Acesso aleatório - cache miss
        for (int idx : indices) {
            if (X[idx][feature_index] <= threshold) {
                left_labels.push_back(y[idx]);
                left_idx.push_back(idx);
            } else {
                right_labels.push_back(y[idx]);
                right_idx.push_back(idx);
            }
        }
        
        if (left_labels.empty() || right_labels.empty()) continue;
        
        double left_gini = calculate_gini(left_labels);
        double right_gini = calculate_gini(right_labels);
        
        double n = static_cast<double>(indices.size());
        double weighted_gini = (left_labels.size() / n) * left_gini +
                               (right_labels.size() / n) * right_gini;
        
        double gain = parent_gini - weighted_gini;
        
        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = threshold;
            best_left = std::move(left_idx);
            best_right = std::move(right_idx);
        }
    }
    
    SplitResult result;
    result.threshold = best_threshold;
    result.gain = best_gain;
    result.left_indices = std::move(best_left);
    result.right_indices = std::move(best_right);
    
    return result;
}

DecisionTree::SplitResult DecisionTree::find_best_split_chunked(
    const std::vector<int>& indices,
    int feature_index,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y) {
    
    // VERSÃO OTIMIZADA - Processamento em BLOCOS DE 100 para cache locality
    std::vector<double> feature_values;
    feature_values.reserve(indices.size());
    
    // Processar em chunks - LOCALIDADE ESPACIAL (CACHE HIT)
    for (size_t i = 0; i < indices.size(); i += CHUNK_SIZE) {
        size_t end = std::min(i + CHUNK_SIZE, indices.size());
        chunk_buffer.clear();
        
        // Carregar chunk completo no buffer (dados contíguos no cache L1/L2)
        for (size_t j = i; j < end; ++j) {
            chunk_buffer.push_back(X[indices[j]][feature_index]);
        }
        
        // Processar chunk enquanto está quente no cache
        for (double val : chunk_buffer) {
            feature_values.push_back(val);
        }
    }
    
    std::vector<double> unique_values = feature_values;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), 
                       unique_values.end());
    
    double best_gain = -1.0;
    double best_threshold = 0.0;
    std::vector<int> best_left, best_right;
    
    // Calcular gini do pai processando em chunks
    std::vector<int> current_labels;
    current_labels.reserve(indices.size());
    
    for (size_t i = 0; i < indices.size(); i += CHUNK_SIZE) {
        size_t end = std::min(i + CHUNK_SIZE, indices.size());
        chunk_labels_buffer.clear();
        
        // Carregar chunk de labels
        for (size_t j = i; j < end; ++j) {
            chunk_labels_buffer.push_back(y[indices[j]]);
        }
        
        // Processar chunk
        for (int label : chunk_labels_buffer) {
            current_labels.push_back(label);
        }
    }
    
    double parent_gini = calculate_gini(current_labels);
    
    // Testar cada threshold processando em chunks
    for (size_t i = 0; i < unique_values.size() - 1; ++i) {
        double threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;
        
        std::vector<int> left_labels, right_labels;
        std::vector<int> left_idx, right_idx;
        
        // Processar em blocos de 100 - melhor uso do cache
        for (size_t chunk_start = 0; chunk_start < indices.size(); chunk_start += CHUNK_SIZE) {
            size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, indices.size());
            
            // Processar bloco completo (localidade temporal)
            for (size_t j = chunk_start; j < chunk_end; ++j) {
                int idx = indices[j];
                if (X[idx][feature_index] <= threshold) {
                    left_labels.push_back(y[idx]);
                    left_idx.push_back(idx);
                } else {
                    right_labels.push_back(y[idx]);
                    right_idx.push_back(idx);
                }
            }
        }
        
        if (left_labels.empty() || right_labels.empty()) continue;
        
        double left_gini = calculate_gini(left_labels);
        double right_gini = calculate_gini(right_labels);
        
        double n = static_cast<double>(indices.size());
        double weighted_gini = (left_labels.size() / n) * left_gini +
                               (right_labels.size() / n) * right_gini;
        
        double gain = parent_gini - weighted_gini;
        
        if (gain > best_gain) {
            best_gain = gain;
            best_threshold = threshold;
            best_left = std::move(left_idx);
            best_right = std::move(right_idx);
        }
    }
    
    SplitResult result;
    result.threshold = best_threshold;
    result.gain = best_gain;
    result.left_indices = std::move(best_left);
    result.right_indices = std::move(best_right);
    
    return result;
}

std::unique_ptr<Node> DecisionTree::build_tree(
    const std::vector<int>& indices,
    const std::vector<std::vector<double>>& X,
    const std::vector<int>& y,
    int depth) {
    
    auto node = std::make_unique<Node>();
    
    std::vector<int> labels;
    labels.reserve(indices.size());
    for (int idx : indices) {
        labels.push_back(y[idx]);
    }
    
    std::map<int, int> class_counts;
    for (int label : labels) {
        class_counts[label]++;
    }
    
    int majority_class = std::max_element(
        class_counts.begin(), class_counts.end(),
        [](const auto& a, const auto& b) { return a.second < b.second; }
    )->first;
    
    if (depth >= max_depth || 
        indices.size() < static_cast<size_t>(min_samples_split) ||
        class_counts.size() == 1) {
        node->is_leaf = true;
        node->predicted_class = majority_class;
        return node;
    }
    
    int num_features = X[0].size();
    double best_gain = -1.0;
    SplitResult best_split;
    int best_feature = -1;
    
    for (int feature = 0; feature < num_features; ++feature) {
        // Escolher versão básica ou com chunks
        auto split = use_chunk_processing ? 
            find_best_split_chunked(indices, feature, X, y) :
            find_best_split_basic(indices, feature, X, y);
            
        if (split.gain > best_gain) {
            best_gain = split.gain;
            best_split = std::move(split);
            best_feature = feature;
        }
    }
    
    if (best_gain <= 0 || best_split.left_indices.empty() || 
        best_split.right_indices.empty()) {
        node->is_leaf = true;
        node->predicted_class = majority_class;
        return node;
    }
    
    node->feature_index = best_feature;
    node->threshold = best_split.threshold;
    node->left = build_tree(best_split.left_indices, X, y, depth + 1);
    node->right = build_tree(best_split.right_indices, X, y, depth + 1);
    
    return node;
}

void DecisionTree::fit(const std::vector<std::vector<double>>& X,
                       const std::vector<int>& y,
                       bool use_chunks) {
    use_chunk_processing = use_chunks;
    
    std::vector<int> all_indices(X.size());
    std::iota(all_indices.begin(), all_indices.end(), 0);
    
    root = build_tree(all_indices, X, y, 0);
}

int DecisionTree::predict_sample(const std::vector<double>& sample, 
                                 const Node* node) const {
    if (node->is_leaf) {
        return node->predicted_class;
    }
    
    if (sample[node->feature_index] <= node->threshold) {
        return predict_sample(sample, node->left.get());
    } else {
        return predict_sample(sample, node->right.get());
    }
}

std::vector<int> DecisionTree::predict(
    const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    
    for (const auto& sample : X) {
        predictions.push_back(predict_sample(sample, root.get()));
    }
    
    return predictions;
}
