#include "RandomForest.h"
#include "DataLoader.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>

std::string get_filename_only(const std::string& path) {
    std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

void train_test_split(const std::vector<std::vector<double>>& X,
                      const std::vector<int>& y,
                      std::vector<std::vector<double>>& X_train,
                      std::vector<int>& y_train,
                      std::vector<std::vector<double>>& X_test,
                      std::vector<int>& y_test,
                      double train_ratio = 0.8) {
    const std::size_t n = X.size();
    std::vector<std::size_t> indices(n);
    for (std::size_t i = 0; i < n; ++i) indices[i] = i;

    // Aleatoriedade determinística para paridade de teste (opcional: tirar seed fixa)
    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    std::size_t n_train = static_cast<std::size_t>(n * train_ratio);

    X_train.clear(); y_train.clear();
    X_test.clear();  y_test.clear();
    X_train.reserve(n_train); y_train.reserve(n_train);
    X_test.reserve(n - n_train); y_test.reserve(n - n_train);

    for (std::size_t i = 0; i < n; ++i) {
        std::size_t idx = indices[i];
        if (i < n_train) {
            X_train.push_back(X[idx]);
            y_train.push_back(y[idx]);
        } else {
            X_test.push_back(X[idx]);
            y_test.push_back(y[idx]);
        }
    }
}

double compute_accuracy(const std::vector<int>& y_true, const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

int main(int argc, char** argv) {
    std::cout << "=== RF Otimizada: LOAD + PREDICAO (SINGLE RUN) ===\n";

    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <dataset.csv> <modelo> [max_samples] [runs_ignored]\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    std::string model_path   = argv[2];
    int max_samples = (argc >= 4) ? std::stoi(argv[3]) : 100000;
    // Ignoramos argv[4] (num_runs) assim como o baseline faz

    std::cout << "Dataset: " << dataset_path << "\n";
    std::cout << "Modelo : " << model_path << "\n";

    // 1. Carregar Dados
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    try {
        DataLoader::load_csv(dataset_path, X, y, max_samples);
        if (X.empty()) return 1;
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << "\n";
        return 1;
    }

    // 2. Split
    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(X, y, X_train, y_train, X_test, y_test, 0.8);
    std::cout << "Amostras Teste: " << X_test.size() << "\n\n";

    // 3. Carregar Modelo
    RandomForest forest(1, 1, 1, 1);
    forest.load_model(model_path);

    // 4. Preparar Memória Flat (Necessário para a otimização funcionar)
    int n_samples = (int)X_test.size();
    int n_features = (int)X_test[0].size();
    std::vector<double> X_flat;
    X_flat.reserve(n_samples * n_features);
    for(const auto& row : X_test) {
        X_flat.insert(X_flat.end(), row.begin(), row.end());
    }

    // 5. Predição (Execução Única)
    std::cout << "Iniciando predicao...\n";
    auto start = std::chrono::high_resolution_clock::now();
    
    std::vector<int> preds = forest.predict_flat(X_flat, n_samples, n_features);
    
    auto end = std::chrono::high_resolution_clock::now();

    // 6. Resultados
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double acc = compute_accuracy(y_test, preds);

    std::cout << "Tempo    : " << ms << " ms\n";
    std::cout << "Acuracia : " << (acc * 100.0) << " %\n";

    // CSV Simples (Append)
    std::string csv_name = "results_predict_optimized_" + get_filename_only(dataset_path) + ".csv";
    std::ofstream csv(csv_name, std::ios::app);
    csv << get_filename_only(dataset_path) << "," << ms << "," << acc << "\n";

    return 0;
}