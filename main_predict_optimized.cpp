#include "RandomForestOptimized.h"
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

    std::mt19937 gen(std::random_device{}());
    std::shuffle(indices.begin(), indices.end(), gen);

    std::size_t n_train = static_cast<std::size_t>(n * train_ratio);

    X_train.clear(); y_train.clear();
    X_test.clear();  y_test.clear();

    X_train.reserve(n_train);
    y_train.reserve(n_train);
    X_test.reserve(n - n_train);
    y_test.reserve(n - n_train);

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

double compute_accuracy(const std::vector<int>& y_true,
                        const std::vector<int>& y_pred) {
    if (y_true.size() != y_pred.size() || y_true.empty()) return 0.0;
    std::size_t correct = 0;
    for (std::size_t i = 0; i < y_true.size(); ++i) {
        if (y_true[i] == y_pred[i]) ++correct;
    }
    return static_cast<double>(correct) / static_cast<double>(y_true.size());
}

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "   Random Forest Otimizada: LOAD + PREDICAO\n";
    std::cout << "========================================================\n\n";

    if (argc < 3) {
        std::cerr << "Uso: " << argv[0]
                  << " <arquivo_dataset.csv> <arquivo_modelo> [max_samples] [num_runs]\n";
        std::cerr << "Exemplo: " << argv[0]
                  << " covertype_dataset.csv optimized_covertype.model 100000 3\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    std::string model_path   = argv[2];

    int max_samples = 100000;
    if (argc >= 4) {
        max_samples = std::stoi(argv[3]);
    }

    int num_runs = 3;
    if (argc >= 5) {
        num_runs = std::stoi(argv[4]);
    }

    std::cout << "Dataset   : " << dataset_path << "\n";
    std::cout << "Modelo    : " << model_path << "\n";
    std::cout << "MaxSamples: " << max_samples << "\n";
    std::cout << "Num runs  : " << num_runs << "\n\n";

    // Carregar dataset
    std::vector<std::vector<double>> X;
    std::vector<int> y;

    std::cout << "Carregando dataset...\n";
    try {
        DataLoader::load_csv(dataset_path, X, y, max_samples);
        if (X.empty()) {
            std::cerr << "❌ Dataset vazio apos carregamento!\n";
            return 1;
        }
        std::cout << "Dataset carregado: " << X.size() << " amostras, "
                  << X[0].size() << " features\n\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Erro ao carregar dataset: " << e.what() << "\n";
        return 1;
    }

    std::vector<std::vector<double>> X_train, X_test;
    std::vector<int> y_train, y_test;
    train_test_split(X, y, X_train, y_train, X_test, y_test, 0.8);

    std::cout << "Treino (nao usado aqui): " << X_train.size() << " amostras\n";
    std::cout << "Teste                  : " << X_test.size()  << " amostras\n\n";

    double total_pred_ms = 0.0;
    double total_acc     = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Iteracao " << (run + 1) << "/" << num_runs << "...\n";

        RandomForestOptimized forest(1, 1, 1, 1); // parametros nao importam para load_model
        std::cout << "  Carregando modelo...\n";
        forest.load_model(model_path);

        std::cout << "  Predizendo em conjunto de teste... ";
        auto start_pred = std::chrono::high_resolution_clock::now();
        std::vector<int> y_pred = forest.predict(X_test);
        auto end_pred   = std::chrono::high_resolution_clock::now();

        double pred_ms =
            std::chrono::duration<double, std::milli>(end_pred - start_pred).count();
        total_pred_ms += pred_ms;
        std::cout << pred_ms << " ms\n";

        double acc = compute_accuracy(y_test, y_pred);
        total_acc += acc;
        std::cout << "  Acuracia: " << std::fixed << std::setprecision(4)
                  << acc * 100.0 << " %\n\n";
    }

    double avg_pred_ms = total_pred_ms / num_runs;
    double avg_acc     = total_acc     / num_runs;

    std::cout << "================= RESULTADOS PREDICAO ===================\n";
    std::cout << std::setw(25) << "Tempo Predicao Medio (ms)"
              << std::setw(20) << std::fixed << std::setprecision(4)
              << avg_pred_ms << "\n";

    std::cout << std::setw(25) << "Acuracia Media (%)"
              << std::setw(20) << std::fixed << std::setprecision(4)
              << (avg_acc * 100.0) << "\n";
    std::cout << "========================================================\n";

    std::string csv_name = "results_predict_optimized_load_" +
                           get_filename_only(dataset_path) + ".csv";
    std::ofstream csv(csv_name);
    csv << "Metodo,Dataset,Modelo,MaxSamples,NumRuns,TempoPredicaoMedio(ms),AcuraciaMedia\n";
    csv << "RandomForestOptimizedPredict,"
        << get_filename_only(dataset_path) << ","
        << model_path << ","
        << max_samples << ","
        << num_runs << ","
        << avg_pred_ms << ","
        << avg_acc << "\n";
    csv.close();

    std::cout << "Resultados salvos em: " << csv_name << "\n";
    return 0;
}
