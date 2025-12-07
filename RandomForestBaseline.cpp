#include "RandomForestBaseline.h"
#include <fstream>
#include <numeric>
#include <random>
#include <algorithm>
#include <unordered_map>

// ============================================================
// Construtor
// ============================================================
RandomForestBaseline::RandomForestBaseline(int n_trees,
                                           int max_depth,
                                           int min_samples_split)
    : n_trees(n_trees),
      max_depth(max_depth),
      min_samples_split(min_samples_split)
{
    trees.reserve(n_trees);
}

// ============================================================
// Bootstrap (amostragem com reposição)
// ============================================================
void RandomForestBaseline::bootstrap_indices(int n_samples,
                                             std::vector<int>& out) const
{
    static thread_local std::mt19937 gen(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, n_samples - 1);

    out.clear();
    out.reserve(n_samples);

    for (int i = 0; i < n_samples; i++)
        out.push_back(dist(gen));
}

// ============================================================
// Treino da floresta
// ============================================================
void RandomForestBaseline::fit(const std::vector<std::vector<double>>& X,
                               const std::vector<int>& y)
{
    trees.clear();
    trees.reserve(n_trees);

    int n_samples = X.size();
    std::vector<int> sample_indices;
    sample_indices.reserve(n_samples);

    for (int t = 0; t < n_trees; t++)
    {
        bootstrap_indices(n_samples, sample_indices);

        // Criar árvore usando índices diretamente (sem copiar dados)
        DecisionTree tree(max_depth, min_samples_split);
        tree.fit(X, y, false, &sample_indices);

        trees.emplace_back(std::move(tree));
    }
}

// ============================================================
// Votação majoritária
// ============================================================
int RandomForestBaseline::majority_vote(const std::vector<int>& votes) const
{
    std::unordered_map<int, int> freq;

    for (int v : votes)
        freq[v]++;

    int best_class = -1;
    int best_count = -1;

    for (auto& kv : freq)
        if (kv.second > best_count) {
            best_class = kv.first;
            best_count = kv.second;
        }

    return best_class;
}

// ============================================================
// Predição
// ============================================================
std::vector<int> RandomForestBaseline::predict(
    const std::vector<std::vector<double>>& X) const
{
    std::vector<int> predictions;
    predictions.reserve(X.size());

    vote_buffer.resize(n_trees);

    for (const auto& sample : X)
    {
        for (int t = 0; t < n_trees; t++)
            vote_buffer[t] = trees[t].predict_one(sample);

        predictions.push_back(majority_vote(vote_buffer));
    }

    return predictions;
}

// ============================================================
// Salvar modelo
// ============================================================
void RandomForestBaseline::save_model(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Erro ao abrir arquivo de modelo para escrita");

    // hiperparâmetros
    out.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));
    out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    out.write(reinterpret_cast<const char*>(&min_samples_split), sizeof(min_samples_split));

    // cada árvore
    for (const auto& tree : trees)
        tree.save_model(out);
}

// ============================================================
// Carregar modelo
// ============================================================
void RandomForestBaseline::load_model(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Erro ao abrir arquivo de modelo para leitura");

    // hiperparâmetros
    in.read(reinterpret_cast<char*>(&n_trees), sizeof(n_trees));
    in.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
    in.read(reinterpret_cast<char*>(&min_samples_split), sizeof(min_samples_split));

    // recriar floresta
    trees.clear();
    trees.reserve(n_trees);

    for (int t = 0; t < n_trees; t++) {
        DecisionTree tree(max_depth, min_samples_split);
        tree.load_model(in);
        trees.emplace_back(std::move(tree)); // ← movimento
    }
}
