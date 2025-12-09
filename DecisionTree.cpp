#include "DecisionTree.h"

#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <limits>
#include <random> // Necessário para sortear as features

// ============================================================
// Construtor (Compatibilidade)
// ============================================================
DecisionTree::DecisionTree(int max_depth, int min_samples_split, int chunk_size)
    : root(nullptr),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      num_classes(0)
{
    (void)chunk_size;
}

// ============================================================
// Movimentação
// ============================================================
DecisionTree::DecisionTree(DecisionTree&& other) noexcept
{
    root = std::move(other.root);
    max_depth = other.max_depth;
    min_samples_split = other.min_samples_split;
    num_classes = other.num_classes;
}

DecisionTree& DecisionTree::operator=(DecisionTree&& other) noexcept
{
    if (this != &other) {
        root = std::move(other.root);
        max_depth = other.max_depth;
        min_samples_split = other.min_samples_split;
        num_classes = other.num_classes;
    }
    return *this;
}

// ============================================================
// FIT
// ============================================================
void DecisionTree::fit(const std::vector<std::vector<double>>& X, 
                       const std::vector<int>& y, 
                       bool use_chunks, 
                       const std::vector<int>* bootstrap_indices) 
{
    (void)use_chunks;
    if (X.empty()) return;

    // 1. Descobrir num_classes
    int max_label = 0;
    for (int label : y) if (label > max_label) max_label = label;
    num_classes = max_label + 1;

    // 2. Transposição Otimizada (Column-Major)
    size_t n_samples = X.size();
    size_t n_features = X[0].size();
    std::vector<std::vector<double>> X_col_major(n_features, std::vector<double>(n_samples));
    
    // Loop blocking para melhorar cache na transposição
    // (Otimização extra caso a matriz seja gigante)
    for (size_t i = 0; i < n_samples; ++i) {
        const double* row_ptr = X[i].data();
        for (size_t j = 0; j < n_features; ++j) {
            X_col_major[j][i] = row_ptr[j];
        }
    }

    // 3. Índices
    std::vector<int> indices;
    if (bootstrap_indices) {
        indices = *bootstrap_indices;
    } else {
        indices.resize(n_samples);
        std::iota(indices.begin(), indices.end(), 0);
    }

    // 4. Construir Recursivamente
    root = build_tree(X_col_major, y, indices, 0);
}

// ============================================================
// BUILD TREE
// ============================================================
std::unique_ptr<Node> DecisionTree::build_tree(
    const std::vector<std::vector<double>>& X_col_major,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int depth)
{
    // Cálculo rápido de pureza
    int majority = -1;
    int max_c = -1;
    
    // Contagem local (Pequena alocação na stack se num_classes for pequeno, 
    // mas aqui usamos heap pelo vetor. Para ultra-perf, usar array fixo se souber classes max)
    std::vector<int> counts(num_classes, 0);
    
    bool is_pure = true;
    int first_label = y[indices[0]];

    for (int idx : indices) {
        int label = y[idx];
        counts[label]++;
        if (label != first_label) is_pure = false;
    }

    // Descobrir majoritária
    for(int c = 0; c < num_classes; c++) {
        if(counts[c] > max_c) {
            max_c = counts[c];
            majority = c;
        }
    }

    // Critérios de Parada (Otimização: Early Exit se for puro)
    if (is_pure || 
        depth >= max_depth || 
        indices.size() < (size_t)min_samples_split) 
    {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority;
        return leaf;
    }

    // Calcular Gini Inicial
    double gini = calculate_gini_from_counts(counts, indices.size());
    if (gini <= 1e-6) { // Praticamente puro
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority;
        return leaf;
    }

    int best_feature = -1;
    double best_threshold = 0.0;
    std::vector<int> left_idx, right_idx;

    find_best_split(X_col_major, y, indices, 
                    best_feature, best_threshold, 
                    left_idx, right_idx, gini);

    if (best_feature == -1 || left_idx.empty() || right_idx.empty()) {
        auto leaf = std::make_unique<Node>();
        leaf->is_leaf = true;
        leaf->predicted_class = majority;
        return leaf;
    }

    auto node = std::make_unique<Node>();
    node->is_leaf = false;
    node->feature_index = best_feature;
    node->threshold = best_threshold;
    node->predicted_class = majority;

    node->left = build_tree(X_col_major, y, left_idx, depth + 1);
    node->right = build_tree(X_col_major, y, right_idx, depth + 1);

    return node;
}

// ============================================================
// FIND BEST SPLIT (A VERSÃO VENCEDORA)
// ============================================================
void DecisionTree::find_best_split(
    const std::vector<std::vector<double>>& X_col_major,
    const std::vector<int>& y,
    const std::vector<int>& indices,
    int& best_feature,
    double& best_threshold,
    std::vector<int>& left_idx,
    std::vector<int>& right_idx,
    double parent_gini)
{
    size_t n_features = X_col_major.size();
    size_t n_samples = indices.size();
    
    double best_gain = -1.0;
    best_feature = -1;

    // --- OTIMIZAÇÃO 1: Feature Subsampling (mtry) ---
    // Em Random Forest, não olhamos todas as features, olhamos sqrt(features).
    // Isso dá um speedup massivo (ex: de 100 colunas para 10).
    size_t n_features_to_check = std::max((size_t)1, (size_t)std::sqrt(n_features));
    
    // Gerador aleatório estático para velocidade (thread_local para thread-safety se necessário)
    static std::mt19937 rng(12345); 
    std::vector<int> feature_candidates(n_features);
    std::iota(feature_candidates.begin(), feature_candidates.end(), 0);
    
    // Embaralha parcial (Fisher-Yates parcial é mais rápido que shuffle total)
    for (size_t i = 0; i < n_features_to_check; ++i) {
        std::uniform_int_distribution<size_t> dist(i, n_features - 1);
        std::swap(feature_candidates[i], feature_candidates[dist(rng)]);
    }

    // --- OTIMIZAÇÃO 2: Hoisting de Memória ---
    // Aloca FORA do loop para reusar a memória em todas as features
    std::vector<SampleEntry> entries(n_samples);
    std::vector<int> total_counts(num_classes, 0);
    std::vector<int> left_counts(num_classes, 0);
    std::vector<int> right_counts(num_classes, 0);

    // Contagem base (uma vez por nó)
    for (int idx : indices) total_counts[y[idx]]++;

    // Loop apenas nas features sorteadas
    for (size_t k = 0; k < n_features_to_check; k++) {
        int f = feature_candidates[k];
        
        // Cópia rápida contígua
        const auto& feature_col = X_col_major[f];
        for (size_t i = 0; i < n_samples; i++) {
            int original_idx = indices[i];
            entries[i].value = feature_col[original_idx];
            entries[i].label = y[original_idx];
            entries[i].original_index = original_idx;
        }

        // Sort (o gargalo aceitável)
        std::sort(entries.begin(), entries.end(), 
            [](const SampleEntry& a, const SampleEntry& b) {
                return a.value < b.value;
            });

        // Reset contadores (sem realocar)
        std::fill(left_counts.begin(), left_counts.end(), 0);
        // Cópia rápida de vetor pequeno
        right_counts = total_counts; 
        
        int n_left = 0;
        int n_right = (int)n_samples;

        // Linear Scan O(N)
        for (size_t i = 0; i < n_samples - 1; i++) {
            int label = entries[i].label;
            
            n_left++;
            n_right--;
            left_counts[label]++;
            right_counts[label]--;

            // Pula duplicatas
            if (entries[i].value == entries[i+1].value) continue;

            // Gini otimizado (inline calculation)
            double gini_left = 1.0;
            double gini_right = 1.0;
            
            for(int c = 0; c < num_classes; c++) {
                if(left_counts[c] > 0) {
                    double p = (double)left_counts[c] / n_left;
                    gini_left -= p*p;
                }
                if(right_counts[c] > 0) {
                    double p = (double)right_counts[c] / n_right;
                    gini_right -= p*p;
                }
            }

            double weighted_gini = ((double)n_left / n_samples) * gini_left + 
                                   ((double)n_right / n_samples) * gini_right;

            double gain = parent_gini - weighted_gini;

            if (gain > best_gain) {
                best_gain = gain;
                best_feature = f;
                best_threshold = (entries[i].value + entries[i+1].value) * 0.5;
            }
        }
    }

    // Reconstrução Final
    if (best_feature != -1) {
        left_idx.reserve(n_samples);
        right_idx.reserve(n_samples);
        const auto& feature_col = X_col_major[best_feature];
        
        // Passada rápida linear usando vetor original
        for (int idx : indices) {
            if (feature_col[idx] <= best_threshold)
                left_idx.push_back(idx);
            else
                right_idx.push_back(idx);
        }
    }
}

// ============================================================
// UTILS
// ============================================================
double DecisionTree::calculate_gini_from_counts(const std::vector<int>& counts, int total) const {
    if (total == 0) return 0.0;
    double impurity = 1.0;
    double inv_total = 1.0 / total; // Multiplicação é mais rápida que divisão
    for (int c : counts) {
        if (c > 0) {
            double p = c * inv_total;
            impurity -= p * p;
        }
    }
    return impurity;
}

int DecisionTree::predict_one(const std::vector<double>& sample) const {
    return predict_sample(sample, root.get());
}

std::vector<int> DecisionTree::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> predictions;
    predictions.reserve(X.size());
    for (const auto& sample : X) {
        predictions.push_back(predict_sample(sample, root.get()));
    }
    return predictions;
}

int DecisionTree::predict_sample(const std::vector<double>& sample, const Node* node) const {
    // Versão iterativa seria mais rápida que recursiva, mas mantemos recursiva pela simplicidade
    if (!node) return -1;
    if (node->is_leaf) return node->predicted_class;

    if (sample[node->feature_index] <= node->threshold)
        return predict_sample(sample, node->left.get());
    else
        return predict_sample(sample, node->right.get());
}

// ============================================================
// SERIALIZATION (BOILERPLATE)
// ============================================================
void DecisionTree::save_model(std::ostream& out) const {
    out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    out.write(reinterpret_cast<const char*>(&min_samples_split), sizeof(min_samples_split));
    out.write(reinterpret_cast<const char*>(&num_classes), sizeof(num_classes));
    save_node(out, root.get());
}

void DecisionTree::save_node(std::ostream& out, const Node* node) const {
    bool exists = (node != nullptr);
    out.write(reinterpret_cast<const char*>(&exists), sizeof(bool));
    if (!exists) return;

    out.write(reinterpret_cast<const char*>(&node->is_leaf), sizeof(bool));
    out.write(reinterpret_cast<const char*>(&node->predicted_class), sizeof(int));
    out.write(reinterpret_cast<const char*>(&node->feature_index), sizeof(int));
    out.write(reinterpret_cast<const char*>(&node->threshold), sizeof(double));

    save_node(out, node->left.get());
    save_node(out, node->right.get());
}

void DecisionTree::load_model(std::istream& in) {
    in.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
    in.read(reinterpret_cast<char*>(&min_samples_split), sizeof(min_samples_split));
    in.read(reinterpret_cast<char*>(&num_classes), sizeof(num_classes));
    root = load_node(in);
}

std::unique_ptr<Node> DecisionTree::load_node(std::istream& in) {
    bool exists;
    in.read(reinterpret_cast<char*>(&exists), sizeof(bool));
    if (!exists) return nullptr;

    auto node = std::make_unique<Node>();
    in.read(reinterpret_cast<char*>(&node->is_leaf), sizeof(bool));
    in.read(reinterpret_cast<char*>(&node->predicted_class), sizeof(int));
    in.read(reinterpret_cast<char*>(&node->feature_index), sizeof(int));
    in.read(reinterpret_cast<char*>(&node->threshold), sizeof(double));

    node->left = load_node(in);
    node->right = load_node(in);
    return node;
}