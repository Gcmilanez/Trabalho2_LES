#include "RandomForestOptimized.h"
#include <fstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <unordered_map>

// ============================================================
// Construtor
// ============================================================
RandomForestOptimized::RandomForestOptimized(int n_trees,
                                             int max_depth,
                                             int min_samples_split,
                                             int chunk_size)
    : n_trees(n_trees),
      max_depth(max_depth),
      min_samples_split(min_samples_split),
      chunk_size(chunk_size)
{
    trees.reserve(n_trees);
}

// ============================================================
// Inicializa ordem base de índices (embaralhados uma vez)
// ============================================================
void RandomForestOptimized::init_base_indices(int n_samples)
{
    base_indices.resize(n_samples);
    std::iota(base_indices.begin(), base_indices.end(), 0);

    static thread_local std::mt19937 gen(std::random_device{}());
    std::shuffle(base_indices.begin(), base_indices.end(), gen);
}

// ============================================================
// Rearranja índices de forma cache-friendly para cada árvore
// ============================================================
void RandomForestOptimized::make_cache_friendly_indices(
    int n_samples,
    int tree_id,
    std::vector<int>& out_indices) const
{
    out_indices.resize(n_samples);

    int offset = (tree_id * chunk_size) % n_samples;

    for (int i = 0; i < n_samples; i++)
        out_indices[i] = base_indices[(i + offset) % n_samples];
}

// ============================================================
// Treino da floresta otimizada
// ============================================================
void RandomForestOptimized::fit(const std::vector<std::vector<double>>& X,
                                const std::vector<int>& y)
{
    const int n_samples = X.size();
    init_base_indices(n_samples);

    trees.clear();
    trees.reserve(n_trees);

    temp_indices.reserve(n_samples);

    for (int t = 0; t < n_trees; t++)
    {
        // reorganiza índices para esta árvore
        make_cache_friendly_indices(n_samples, t, temp_indices);

        // cria a árvore com chunk_size configurado corretamente
        DecisionTree tree(max_depth, min_samples_split, chunk_size);
        tree.fit(X, y, true); // ← usa chunked = otimizado

        trees.emplace_back(std::move(tree)); // ← movimento, não cópia
    }
}

// ============================================================
// Votação majoritária
// ============================================================
int RandomForestOptimized::majority_vote(const std::vector<int>& votes) const
{
    std::unordered_map<int,int> freq;

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
std::vector<int> RandomForestOptimized::predict(
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
// Salvamento do modelo completo
// ============================================================
void RandomForestOptimized::save_model(const std::string& filename) const
{
    std::ofstream out(filename, std::ios::binary);
    if (!out)
        throw std::runtime_error("Erro ao abrir arquivo de modelo otimizado para escrita.");

    // hiperparâmetros
    out.write(reinterpret_cast<const char*>(&n_trees), sizeof(n_trees));
    out.write(reinterpret_cast<const char*>(&max_depth), sizeof(max_depth));
    out.write(reinterpret_cast<const char*>(&min_samples_split), sizeof(min_samples_split));
    out.write(reinterpret_cast<const char*>(&chunk_size), sizeof(chunk_size));

    // salvar cada árvore
    for (const auto& tree : trees)
        tree.save_model(out);
}

// ============================================================
// Carregamento do modelo completo
// ============================================================
void RandomForestOptimized::load_model(const std::string& filename)
{
    std::ifstream in(filename, std::ios::binary);
    if (!in)
        throw std::runtime_error("Erro ao abrir arquivo de modelo otimizado para leitura.");

    // hiperparâmetros
    in.read(reinterpret_cast<char*>(&n_trees), sizeof(n_trees));
    in.read(reinterpret_cast<char*>(&max_depth), sizeof(max_depth));
    in.read(reinterpret_cast<char*>(&min_samples_split), sizeof(min_samples_split));
    in.read(reinterpret_cast<char*>(&chunk_size), sizeof(chunk_size));

    // recriar floresta
    trees.clear();
    trees.reserve(n_trees);

    for (int t = 0; t < n_trees; t++)
    {
        DecisionTree tree(max_depth, min_samples_split, chunk_size);
        tree.load_model(in);
        trees.emplace_back(std::move(tree)); // ← movimento, não cópia
    }
}
