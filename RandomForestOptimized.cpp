// RandomForestOptimized.cpp
#include "RandomForestOptimized.h"
#include <random>
#include <numeric>
#include <algorithm>

RandomForestOptimized::RandomForestOptimized(
    int n_trees_,
    int max_depth_,
    int min_samples_split_,
    int chunk_size_
) : n_trees(n_trees_),
    max_depth(max_depth_),
    min_samples_split(min_samples_split_),
    chunk_size(chunk_size_) {

    trees.reserve(n_trees);
}

void RandomForestOptimized::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y) {
    const int n_samples = static_cast<int>(X.size());
    if (n_samples == 0) return;

    trees.clear();

    init_base_indices(n_samples);
    indices_buffer.reserve(n_samples);

    for (int t = 0; t < n_trees; ++t) {
        // gera índices para esta árvore de forma mais cache-friendly
        make_cache_friendly_indices(n_samples, t, indices_buffer);

        // **versão inicial simples**:
        // ainda copia os dados, mas em blocos contíguos da base embaralhada
        std::vector<std::vector<double>> X_sub;
        std::vector<int> y_sub;
        X_sub.reserve(indices_buffer.size());
        y_sub.reserve(indices_buffer.size());

        for (int idx : indices_buffer) {
            X_sub.push_back(X[idx]);
            y_sub.push_back(y[idx]);
        }

        DecisionTree tree;
        // TODO: ligar o modo de chunk da sua DecisionTree.
        // Algo do tipo:
        // tree.set_max_depth(max_depth);
        // tree.set_min_samples_split(min_samples_split);
        // tree.set_chunk_size(chunk_size);    // se existir
        // tree.fit(X_sub, y_sub, /*use_chunks=*/true);

        trees.push_back(std::move(tree));
    }
}

std::vector<int> RandomForestOptimized::predict(
    const std::vector<std::vector<double>>& X) const
{
    std::vector<int> preds;
    preds.reserve(X.size());

    vote_buffer.resize(trees.size());

    for (const auto& sample : X) {
        for (size_t i = 0; i < trees.size(); ++i) {
            // TODO: adaptar à API real
            std::vector<std::vector<double>> tmp = { sample };
            int pred = trees[i].predict(tmp)[0];
            vote_buffer[i] = pred;
        }
        preds.push_back(majority_vote(vote_buffer));
    }

    return preds;
}

int RandomForestOptimized::predict_one(const std::vector<double>& x) const {
    vote_buffer.resize(trees.size());
    for (size_t i = 0; i < trees.size(); ++i) {
        // TODO: adaptar
        std::vector<std::vector<double>> tmp = { x };
        int pred = trees[i].predict(tmp)[0];
        vote_buffer[i] = pred;
    }
    return majority_vote(vote_buffer);
}

void RandomForestOptimized::init_base_indices(int n_samples) {
    base_indices.resize(n_samples);
    std::iota(base_indices.begin(), base_indices.end(), 0);

    // embaralha uma vez só
    static thread_local std::mt19937 gen(std::random_device{}());
    std::shuffle(base_indices.begin(), base_indices.end(), gen);
}

void RandomForestOptimized::make_cache_friendly_indices(
    int n_samples,
    int tree_id,
    std::vector<int>& out_indices) const
{
    out_indices.clear();

    // Exemplo simples: pega todos em ordem circular mas com offset diferente
    // para cada árvore (mais cache-friendly que sorteio 100% aleatório).
    int start = (tree_id * n_samples / 3) % n_samples;

    out_indices.reserve(n_samples);
    for (int i = 0; i < n_samples; ++i) {
        int idx = base_indices[(start + i) % n_samples];
        out_indices.push_back(idx);
    }

    // Se quiser menos que n_samples por árvore, reduza aqui
    // ex.: out_indices.resize(n_samples * 0.8);
}

int RandomForestOptimized::majority_vote(const std::vector<int>& votes) const {
    if (votes.empty()) return -1;

    std::vector<int> sorted = votes;
    std::sort(sorted.begin(), sorted.end());

    int best_class = sorted[0];
    int best_count = 1;
    int current_class = sorted[0];
    int current_count = 1;

    for (size_t i = 1; i < sorted.size(); ++i) {
        if (sorted[i] == current_class) {
            current_count++;
        } else {
            if (current_count > best_count) {
                best_count = current_count;
                best_class = current_class;
            }
            current_class = sorted[i];
            current_count = 1;
        }
    }
    if (current_count > best_count) {
        best_class = current_class;
    }

    return best_class;
}
