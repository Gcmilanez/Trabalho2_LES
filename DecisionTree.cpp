#include "DecisionTree.h"
#include <algorithm>
#include <numeric>
#include <map>
#include <limits>

DecisionTree::DecisionTree(int max_depth, int min_samples_split)
    : max_depth(max_depth), min_samples_split(min_samples_split),
      use_chunk_processing(false), cache_friendly_accesses(0), random_accesses(0) {
    // Pré-alocação de buffers para versão com chunks
    // Evita realocações dinâmicas durante o processamento
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
    
    // ============================================================
    // VERSÃO GENÉRICA - Acesso aleatório aos dados
    // ============================================================
    // PROBLEMA DE CACHE:
    // 1. indices[] pode apontar para posições distantes em X
    // 2. X[idx] pode estar em linhas de cache diferentes
    // 3. Cada acesso pode causar cache miss (~200 ciclos de latência)
    // 4. CPU fica ociosa esperando dados da RAM
    // ============================================================
    
    std::vector<double> feature_values;
    feature_values.reserve(indices.size());
    
    // ACESSO NÃO SEQUENCIAL - Alta probabilidade de cache miss
    for (int idx : indices) {
        feature_values.push_back(X[idx][feature_index]);
        // Não incrementar contador aqui - só contar nos splits
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
        
        // ACESSO ALEATÓRIO - cache miss em cada iteração
        for (int idx : indices) {
            random_accesses++; // Contador
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
    
    // ============================================================
    // VERSÃO OTIMIZADA - CHUNKS de 100 para LOCALIDADE TEMPORAL
    // ============================================================
    // Processa dados em blocos de 100 elementos:
    // 1. Carrega 100 elementos no cache
    // 2. Processa IMEDIATAMENTE enquanto estão "quentes"
    // 3. Passa para próximo chunk
    // Benefício: Dados permanecem no cache L1/L2 durante processamento
    // ============================================================
    
    std::vector<double> feature_values;
    feature_values.reserve(indices.size());
    
    std::vector<int> current_labels;
    current_labels.reserve(indices.size());
    
    // PRÉ-COMPUTAR em CHUNKS de 100 (localidade temporal)
    for (size_t i = 0; i < indices.size(); i += CHUNK_SIZE) {
        size_t end = std::min(i + CHUNK_SIZE, indices.size());
        
        // Carregar CHUNK de 100 elementos
        for (size_t j = i; j < end; ++j) {
            int idx = indices[j];
            feature_values.push_back(X[idx][feature_index]);
            current_labels.push_back(y[idx]);
        }
        // Dados estão "quentes" no cache, próximo chunk continuará...
    }
    
    std::vector<double> unique_values = feature_values;
    std::sort(unique_values.begin(), unique_values.end());
    unique_values.erase(std::unique(unique_values.begin(), unique_values.end()), 
                       unique_values.end());
    
    double best_gain = -1.0;
    double best_threshold = 0.0;
    std::vector<int> best_left, best_right;
    double parent_gini = calculate_gini(current_labels);
    
    // PROCESSAR thresholds usando dados pré-computados (já no cache)
    for (size_t i = 0; i < unique_values.size() - 1; ++i) {
        double threshold = (unique_values[i] + unique_values[i + 1]) / 2.0;
        
        std::vector<int> left_labels, right_labels;
        std::vector<int> left_idx, right_idx;
        
        // Acessar em CHUNKS de 100 (localidade temporal)
        for (size_t chunk_start = 0; chunk_start < feature_values.size(); chunk_start += CHUNK_SIZE) {
            size_t chunk_end = std::min(chunk_start + CHUNK_SIZE, feature_values.size());
            
            // Processar CHUNK completo enquanto está no cache
            for (size_t j = chunk_start; j < chunk_end; ++j) {
                cache_friendly_accesses++;
                if (feature_values[j] <= threshold) {
                    left_labels.push_back(current_labels[j]);
                    left_idx.push_back(indices[j]);
                } else {
                    right_labels.push_back(current_labels[j]);
                    right_idx.push_back(indices[j]);
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
