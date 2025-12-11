#include "DecisionTree.h"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <cstring>

DecisionTree::DecisionTree(int max_depth, int min_samples_split, int chunk_size)
    : root(nullptr), is_flat_built(false), max_depth(max_depth), 
      min_samples_split(min_samples_split), chunk_size(chunk_size), use_optimized_mode(false) {}

void DecisionTree::fit_baseline(const std::vector<double>& X_flat, int n_samples, int n_features, const std::vector<int>& y, const std::vector<int>& indices) {
    this->use_optimized_mode = false;
    sort_buffer.reserve(n_samples); 
    root = build_tree(&X_flat, n_samples, n_features, y, indices, 0);
    is_flat_built = false; flat_tree.clear();
}

void DecisionTree::fit_optimized(const std::vector<double>& X_flat, int n_samples, int n_features, const std::vector<int>& y, const std::vector<int>& indices) {
    this->use_optimized_mode = true;
    sort_buffer.reserve(n_samples); 
    root = build_tree(&X_flat, n_samples, n_features, y, indices, 0);
    flatten_tree(); 
}

void DecisionTree::flatten_tree() {
    if (!root) return;
    flat_tree.clear();
    flat_tree.reserve(std::pow(2, std::min(max_depth + 1, 20))); 
    fill_flat_tree(root.get());
    is_flat_built = true;
}

int DecisionTree::fill_flat_tree(const Node* node) {
    if (!node) return -1;
    int current_idx = (int)flat_tree.size();
    flat_tree.emplace_back();

    flat_tree[current_idx].predicted_class = node->predicted_class;
    flat_tree[current_idx].threshold = node->threshold;
    flat_tree[current_idx].feature_index = node->is_leaf ? -1 : node->feature_index;
    flat_tree[current_idx].right_child_offset = 0; 

    if (!node->is_leaf) {
        fill_flat_tree(node->left.get());
        int right_absolute_idx = fill_flat_tree(node->right.get());
        flat_tree[current_idx].right_child_offset = right_absolute_idx - current_idx;
    }
    return current_idx;
}

int DecisionTree::predict_sample_flat_ptr(const double* sample) const {
    const FlatNode* node = flat_tree.data();

    // Loop sem verificação de limites (seguro se a lógica da árvore estiver correta)
    while (node->feature_index >= 0) {
        // Acesso direto à memória: sample[idx]
        // Branchless select para o próximo nó
        int go_right = (sample[node->feature_index] > node->threshold);
        node += (go_right ? node->right_child_offset : 1);
    }
    return node->predicted_class;
}

// Otimizado para struct de 32 bytes alinhada
int DecisionTree::predict_sample_flat(const std::vector<double>& sample) const {
    const FlatNode* node = flat_tree.data();

    while (node->feature_index >= 0) {
        // Carrega double direto (sem cast)
        double val = sample[node->feature_index];
        
        // Branchless
        int go_right = (val > node->threshold);
        int offset = go_right ? node->right_child_offset : 1;
        
        node += offset;
    }
    return node->predicted_class;
}

int DecisionTree::predict_one(const std::vector<double>& sample) const {
    if (use_optimized_mode && is_flat_built && !flat_tree.empty()) {
        return predict_sample_flat(sample);
    } 
    return predict_sample_ptr(sample, root.get());
}

// ... [RESTO DOS MÉTODOS IGUAIS] ...
int DecisionTree::predict_sample_ptr(const std::vector<double>& sample, const Node* node) const {
    if (!node || node->is_leaf) return node ? node->predicted_class : -1;
    if (sample[node->feature_index] <= node->threshold) 
        return predict_sample_ptr(sample, node->left.get());
    else 
        return predict_sample_ptr(sample, node->right.get());
}

std::vector<int> DecisionTree::predict(const std::vector<std::vector<double>>& X) const {
    std::vector<int> res; res.reserve(X.size());
    for(const auto& s : X) res.push_back(predict_one(s));
    return res;
}

std::unique_ptr<Node> DecisionTree::build_tree(const std::vector<double>* X_flat, int n_total_samples, int n_features, const std::vector<int>& y, const std::vector<int>& indices, int depth) {
    // [MANTER O CÓDIGO DE BUILD_TREE DA RESPOSTA ANTERIOR]
    // Apenas resumindo aqui para caber na resposta, use o código completo que você já tem
    std::vector<int> current_labels; current_labels.reserve(indices.size());
    for (int idx : indices) current_labels.push_back(y[idx]);
    double gini = calculate_gini(current_labels);
    if (depth >= max_depth || indices.size() < (size_t)min_samples_split || gini == 0.0) {
        auto leaf = std::make_unique<Node>(); leaf->is_leaf = true; leaf->predicted_class = majority_class(current_labels); return leaf;
    }
    int best_feature = -1; double best_threshold = 0.0;
    std::vector<int> left_idx, right_idx;
    if (use_optimized_mode) find_best_split_optimized(*X_flat, n_total_samples, n_features, y, indices, best_feature, best_threshold, left_idx, right_idx, gini);
    else find_best_split_naive(*X_flat, n_total_samples, n_features, y, indices, best_feature, best_threshold, left_idx, right_idx, gini);
    if (best_feature == -1) {
        auto leaf = std::make_unique<Node>(); leaf->is_leaf = true; leaf->predicted_class = majority_class(current_labels); return leaf;
    }
    auto node = std::make_unique<Node>(); node->is_leaf = false; node->feature_index = best_feature; node->threshold = best_threshold;
    node->left = build_tree(X_flat, n_total_samples, n_features, y, left_idx, depth + 1);
    node->right = build_tree(X_flat, n_total_samples, n_features, y, right_idx, depth + 1);
    return node;
}

// [MANTER find_best_split_naive e optimized, calculate_gini, majority_class, save_model, save_node]
// ...
void DecisionTree::find_best_split_naive(const std::vector<double>& X_flat, int n, int f, const std::vector<int>& y, const std::vector<int>& idx, int& bf, double& bt, std::vector<int>& l, std::vector<int>& r, double pg) {
    const size_t n_node_samples = idx.size(); double best_gain = -1.0;
    int n_subset = (int)std::sqrt(f); if(n_subset<1) n_subset=1;
    std::vector<int> feature_indices(f); std::iota(feature_indices.begin(), feature_indices.end(), 0);
    static thread_local std::mt19937 gen(std::random_device{}()); std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    int max_label = 0; for(int i : idx) if(y[i]>max_label) max_label=y[i];
    std::vector<int> rc(max_label+1), lc(max_label+1);

    for (int k=0; k<n_subset; ++k) {
        int feat = feature_indices[k];
        sort_buffer.clear(); const double* ptr = &X_flat[feat * n];
        for (int i : idx) sort_buffer.push_back({ptr[i], i});
        std::sort(sort_buffer.begin(), sort_buffer.end());
        std::fill(lc.begin(), lc.end(), 0); std::fill(rc.begin(), rc.end(), 0);
        for(int i:idx) rc[y[i]]++;
        double sr=0; for(int c:rc) if(c>0) sr+=(double)c*c;
        double sl=0, size_l=0, size_r=(double)n_node_samples;
        
        for(size_t i=0; i<n_node_samples-1; ++i) {
             int id=sort_buffer[i].second; int lbl=y[id];
             int cr=rc[lbl]; sr-= (double)cr*cr; rc[lbl]--; sr+= (double)rc[lbl]*rc[lbl];
             int cl=lc[lbl]; sl-= (double)cl*cl; lc[lbl]++; sl+= (double)lc[lbl]*lc[lbl];
             size_l++; size_r--;
             if(sort_buffer[i].first == sort_buffer[i+1].first) continue;
             double gl = 1.0-(sl/(size_l*size_l)); double gr = 1.0-(sr/(size_r*size_r));
             double gain = pg - (size_l/n_node_samples*gl) - (size_r/n_node_samples*gr);
             if(gain>best_gain) { best_gain=gain; bf=feat; bt=(sort_buffer[i].first+sort_buffer[i+1].first)/2.0; }
        }
    }
    if(best_gain>0) {
        const double* ptr = &X_flat[bf*n];
        l.reserve(n_node_samples); r.reserve(n_node_samples);
        for(int i:idx) { if(ptr[i]<=bt) l.push_back(i); else r.push_back(i); }
    }
}
void DecisionTree::find_best_split_optimized(const std::vector<double>& X_flat, int n, int f, const std::vector<int>& y, const std::vector<int>& idx, int& bf, double& bt, std::vector<int>& l, std::vector<int>& r, double pg) {
    double best_gain = -1.0; size_t n_node_samples = idx.size();
    int n_subset = (int)std::sqrt(f); if(n_subset<1) n_subset=1;
    std::vector<int> feature_indices(f); std::iota(feature_indices.begin(), feature_indices.end(), 0);
    static thread_local std::mt19937 gen(std::random_device{}()); std::shuffle(feature_indices.begin(), feature_indices.end(), gen);
    int max_label = 0; for(int i : idx) if(y[i]>max_label) max_label=y[i];
    std::vector<int> rc(max_label+1), lc(max_label+1);

    for (int k=0; k<n_subset; ++k) {
        int feat = feature_indices[k];
        sort_buffer.clear(); const double* ptr = &X_flat[feat * n];
        for(size_t cs=0; cs<n_node_samples; cs+=chunk_size) {
            size_t ce = std::min(cs+(size_t)chunk_size, n_node_samples);
            for(size_t i=cs; i<ce; ++i) { int id=idx[i]; sort_buffer.push_back({ptr[id], id}); }
        }
        std::sort(sort_buffer.begin(), sort_buffer.end());
        std::fill(lc.begin(), lc.end(), 0); std::fill(rc.begin(), rc.end(), 0);
        for(int i:idx) rc[y[i]]++;
        double sr=0; for(int c:rc) if(c>0) sr+=(double)c*c;
        double sl=0, size_l=0, size_r=(double)n_node_samples;
        
        for(size_t i=0; i<n_node_samples-1; ++i) {
             int id=sort_buffer[i].second; int lbl=y[id];
             int cr=rc[lbl]; sr-= (double)cr*cr; rc[lbl]--; sr+= (double)rc[lbl]*rc[lbl];
             int cl=lc[lbl]; sl-= (double)cl*cl; lc[lbl]++; sl+= (double)lc[lbl]*lc[lbl];
             size_l++; size_r--;
             if(sort_buffer[i].first == sort_buffer[i+1].first) continue;
             double gl = 1.0-(sl/(size_l*size_l)); double gr = 1.0-(sr/(size_r*size_r));
             double gain = pg - (size_l/n_node_samples*gl) - (size_r/n_node_samples*gr);
             if(gain>best_gain) { best_gain=gain; bf=feat; bt=(sort_buffer[i].first+sort_buffer[i+1].first)/2.0; }
        }
    }
    if(best_gain>0) {
        const double* ptr = &X_flat[bf*n];
        l.reserve(n_node_samples); r.reserve(n_node_samples);
        for(int i:idx) { if(ptr[i]<=bt) l.push_back(i); else r.push_back(i); }
    }
}
double DecisionTree::calculate_gini(const std::vector<int>& labels) const {
    if (labels.empty()) return 0.0;
    std::unordered_map<int, int> counts; for (int l : labels) counts[l]++;
    double imp = 1.0; double n = (double)labels.size();
    for (auto& kv : counts) imp -= pow(kv.second / n, 2);
    return imp;
}
int DecisionTree::majority_class(const std::vector<int>& labels) const {
    if (labels.empty()) return -1;
    std::unordered_map<int, int> counts; int best_c = -1, best_cnt = -1;
    for (int l : labels) { counts[l]++; if (counts[l] > best_cnt) { best_cnt = counts[l]; best_c = l; } }
    return best_c;
}
void DecisionTree::save_model(std::ostream& out) const {
    out.write((char*)&max_depth, sizeof(int));
    out.write((char*)&min_samples_split, sizeof(int));
    out.write((char*)&chunk_size, sizeof(int));
    out.write((char*)&use_optimized_mode, sizeof(bool));
    save_node(out, root.get());
}
void DecisionTree::save_node(std::ostream& out, const Node* node) const {
    bool exists = (node != nullptr); out.write((char*)&exists, sizeof(bool));
    if(!exists) return;
    out.write((char*)&node->is_leaf, sizeof(bool));
    out.write((char*)&node->feature_index, sizeof(int));
    out.write((char*)&node->threshold, sizeof(double));
    out.write((char*)&node->predicted_class, sizeof(int));
    save_node(out, node->left.get()); save_node(out, node->right.get());
}
void DecisionTree::load_model(std::istream& in) {
    in.read((char*)&max_depth, sizeof(int)); in.read((char*)&min_samples_split, sizeof(int));
    in.read((char*)&chunk_size, sizeof(int)); in.read((char*)&use_optimized_mode, sizeof(bool));
    root = load_node(in);
    if (use_optimized_mode) flatten_tree(); else { is_flat_built = false; flat_tree.clear(); }
}
std::unique_ptr<Node> DecisionTree::load_node(std::istream& in) {
    bool ex; in.read((char*)&ex, sizeof(bool)); if(!ex) return nullptr;
    auto node = std::make_unique<Node>();
    in.read((char*)&node->is_leaf, sizeof(bool)); in.read((char*)&node->feature_index, sizeof(int));
    in.read((char*)&node->threshold, sizeof(double)); in.read((char*)&node->predicted_class, sizeof(int));
    node->left = load_node(in); node->right = load_node(in);
    return node;
}