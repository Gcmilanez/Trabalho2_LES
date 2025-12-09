#include "RandomForest.h" // Classe unificada
#include "DataLoader.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>
#include <random>
#include <algorithm>

// Função auxiliar para dividir treino/teste
void train_test_split(const std::vector<std::vector<double>>& X, const std::vector<int>& y,
                      std::vector<std::vector<double>>& X_test, std::vector<int>& y_test,
                      double train_ratio = 0.8) {
    size_t n = X.size();
    size_t n_train = (size_t)(n * train_ratio);
    // Para simplificar e garantir determinismo no teste, pegamos o final do dataset
    // (Em produção faríamos shuffle, mas para benchmark de velocidade de predição isso basta)
    for(size_t i = n_train; i < n; ++i) {
        X_test.push_back(X[i]);
        y_test.push_back(y[i]);
    }
}

double compute_accuracy(const std::vector<int>& true_y, const std::vector<int>& pred_y) {
    if(true_y.empty()) return 0.0;
    int correct = 0;
    for(size_t i=0; i<true_y.size(); ++i) {
        if(true_y[i] == pred_y[i]) correct++;
    }
    return (double)correct / true_y.size();
}

int main(int argc, char** argv) {
    std::cout << "=== Random Forest Baseline: LOAD + PREDICT ===\n";

    if (argc < 3) {
        std::cerr << "Uso: " << argv[0] << " <dataset.csv> <model_path> [max_samples]\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    std::string model_path = argv[2];
    int max_samples = (argc >= 4) ? std::stoi(argv[3]) : 100000;

    std::vector<std::vector<double>> X, X_test;
    std::vector<int> y, y_test;
    
    try {
        DataLoader::load_csv(dataset_path, X, y, max_samples);
        // Cria set de teste (20% final)
        std::vector<int> dummy_y; std::vector<std::vector<double>> dummy_X;
        train_test_split(X, y, X_test, y_test); 
        std::cout << "Amostras de Teste: " << X_test.size() << "\n";
    } catch (...) { return 1; }

    // Carregar Modelo
    // Parâmetros no construtor são dummy, pois load_model vai sobrescrever
    RandomForest forest(1, 1, 1); 
    std::cout << "Carregando modelo: " << model_path << "\n";
    forest.load_model(model_path); // Certifique-se de implementar load_model na RandomForest

    // Benchmark Predição
    auto start = std::chrono::high_resolution_clock::now();
    std::vector<int> preds = forest.predict(X_test);
    auto end = std::chrono::high_resolution_clock::now();
    
    double ms = std::chrono::duration<double, std::milli>(end - start).count();
    double acc = compute_accuracy(y_test, preds);

    std::cout << "Tempo Predicao: " << ms << " ms\n";
    std::cout << "Acuracia: " << (acc * 100.0) << "%\n";

    return 0;
}