#include "DecisionTree.h"
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>

struct BenchmarkResult {
    double time_basic;
    double time_chunked;
    double speedup;
    int dataset_size;
    int num_features;
};

std::vector<std::vector<double>> generate_dataset(int n_samples, int n_features, 
                                                   std::mt19937& rng) {
    std::uniform_real_distribution<double> dist(0.0, 100.0);
    std::vector<std::vector<double>> X(n_samples, std::vector<double>(n_features));
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X[i][j] = dist(rng);
        }
    }
    
    return X;
}

std::vector<int> generate_labels(int n_samples, int n_classes, std::mt19937& rng) {
    std::uniform_int_distribution<int> dist(0, n_classes - 1);
    std::vector<int> y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        y[i] = dist(rng);
    }
    
    return y;
}

BenchmarkResult run_benchmark(int n_samples, int n_features) {
    std::mt19937 rng(42);
    
    auto X = generate_dataset(n_samples, n_features, rng);
    auto y = generate_labels(n_samples, 3, rng);
    
    // 1. Benchmark BÁSICO (sem chunks - acesso aleatório)
    DecisionTree tree_basic(5, 2);
    auto start = std::chrono::high_resolution_clock::now();
    tree_basic.fit(X, y, false);
    auto end = std::chrono::high_resolution_clock::now();
    double time_basic = std::chrono::duration<double, std::milli>(end - start).count();
    
    // 2. Benchmark COM CHUNKS (processamento em blocos de 100)
    DecisionTree tree_chunked(5, 2);
    start = std::chrono::high_resolution_clock::now();
    tree_chunked.fit(X, y, true);
    end = std::chrono::high_resolution_clock::now();
    double time_chunked = std::chrono::duration<double, std::milli>(end - start).count();
    
    BenchmarkResult result;
    result.time_basic = time_basic;
    result.time_chunked = time_chunked;
    result.speedup = time_basic / time_chunked;
    result.dataset_size = n_samples;
    result.num_features = n_features;
    
    return result;
}

void save_results_csv(const std::vector<BenchmarkResult>& results, 
                      const std::string& filename) {
    std::ofstream file(filename);
    file << "Dataset Size,Features,Time Basic (ms),Time Chunked (ms),Speedup\n";
    
    for (const auto& result : results) {
        file << result.dataset_size << ","
             << result.num_features << ","
             << std::fixed << std::setprecision(2) 
             << result.time_basic << ","
             << result.time_chunked << ","
             << result.speedup << "\n";
    }
    
    file.close();
}

int main() {
    std::cout << "=== Decision Tree Benchmark ===\n";
    std::cout << "Comparacao: Basica vs Com Blocos de Cache (Chunks de 100)\n\n";
    
    std::vector<BenchmarkResult> results;
    std::vector<int> dataset_sizes = {100, 500, 1000, 2000, 5000};
    std::vector<int> feature_counts = {5, 10, 15};
    
    std::cout << std::setw(12) << "Size"
              << std::setw(10) << "Features"
              << std::setw(18) << "Basica (ms)"
              << std::setw(18) << "Com Chunks (ms)"
              << std::setw(12) << "Speedup\n";
    std::cout << std::string(70, '-') << "\n";
    
    for (int n_samples : dataset_sizes) {
        for (int n_features : feature_counts) {
            std::cout << "Executando: " << n_samples << " samples, " 
                      << n_features << " features..." << std::flush;
            
            auto result = run_benchmark(n_samples, n_features);
            results.push_back(result);
            
            std::cout << "\r" << std::setw(12) << result.dataset_size
                      << std::setw(10) << result.num_features
                      << std::setw(18) << std::fixed << std::setprecision(2) 
                      << result.time_basic
                      << std::setw(18) << result.time_chunked
                      << std::setw(12) << std::setprecision(2) 
                      << result.speedup << "x\n";
        }
    }
    
    std::cout << "\n=== Salvando resultados ===\n";
    save_results_csv(results, "benchmark_results.csv");
    std::cout << "Resultados salvos em: benchmark_results.csv\n";
    
    // Estatísticas
    double avg_speedup = 0.0;
    double max_speedup = 0.0;
    for (const auto& result : results) {
        avg_speedup += result.speedup;
        if (result.speedup > max_speedup) {
            max_speedup = result.speedup;
        }
    }
    avg_speedup /= results.size();
    
    std::cout << "\n=== Resumo ===\n";
    std::cout << "Speedup medio: " << std::fixed << std::setprecision(2) 
              << avg_speedup << "x\n";
    std::cout << "Speedup maximo: " << max_speedup << "x\n";
    std::cout << "\nOtimizacao: Processamento em blocos de 100 elementos\n";
    std::cout << "Beneficio: Melhor localidade espacial e temporal do cache\n";
    
    return 0;
}
