// RandomForestBaseline.cpp
#include "RandomForestBaseline.h"
#include <random>
#include <numeric>  // iota
#include <algorithm> // max_element

RandomForestBaseline::RandomForestBaseline(
    int n_trees_,
    int max_depth_,
    int min_samples_split_
) : n_trees(n_trees_),
    max_depth(max_depth_),
    min_samples_split(min_samples_split_) {

    trees.reserve(n_trees);
}

void RandomForestBaseline::fit(const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y) {
    const int n_samples = static_cast<int>(X.size());
    if (n_samples == 0) return;

    std::vector<int> indices;
    indices.reserve(n_samples);

    trees.clear();

    for (int t = 0; t < n_trees; ++t) {
        bootstrap_indices(n_samples, indices);

        // monta subconjunto de dados (implementação simples e "comum")
        std::vector<std::vector<double>> X_boot;
        std::vector<int> y_boot;
        X_boot.reserve(n_samples);
        y_boot.reserve(n_samples);

        for (int idx : indices) {
            X_boot.push_back(X[idx]);
            y_boot.push_back(y[idx]);
        }

        DecisionTree tree;
        // TODO: ajustar para a API real da sua DecisionTree:
        // por ex:
        // tree.set_max_depth(max_depth);
        // tree.set_min_samples_split(min_samples_split);
        // tree.fit(X_boot, y_boot, /*use_chunks=*/false);

        trees.push_back(std::move(tree));
    }
}

std::vector<int> RandomForestBaseline::predict(
    const std::vector<std::vector<double>>& X) const
{
    std::vector<int> preds;
    preds.reserve(X.size());

    vote_buffer.resize(trees.size());

    for (const auto& sample : X) {
        // predição de todas as árvores
        for (size_t i = 0; i < trees.size(); ++i) {
            // TODO: adaptar à API real de predict da sua árvore
            // Exemplo genérico:
            std::vector<std::vector<double>> tmp = { sample };
            int pred = trees[i].predict(tmp)[0];
            vote_buffer[i] = pred;
        }
        preds.push_back(majority_vote(vote_buffer));
    }

    return preds;
}

int RandomForestBaseline::predict_one(const std::vector<double>& x) const {
    vote_buffer.resize(trees.size());

    for (size_t i = 0; i < trees.size(); ++i) {
        // TODO: adaptar à API real
        std::vector<std::vector<double>> tmp = { x };
        int pred = trees[i].predict(tmp)[0];
        vote_buffer[i] = pred;
    }

    return majority_vote(vote_buffer);
}

void RandomForestBaseline::bootstrap_indices(int n_samples,
                                             std::vector<int>& indices) const {
    indices.clear();
    indices.reserve(n_samples);

    // RNG simples; você pode fixar seed se quiser reprodutibilidade
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n_samples - 1);

    for (int i = 0; i < n_samples; ++i) {
        indices.push_back(dist(gen));
    }
}

int RandomForestBaseline::majority_vote(const std::vector<int>& votes) const {
    if (votes.empty()) return -1;

    // conta frequência simples
    // (se o número de classes for pequeno, dá pra otimizar depois)
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
