#include "DecisionTree.h"
#include "DataLoader.h"
#include <iostream>
#include <chrono>
#include <random>
#include <fstream>
#include <iomanip>

int main() {
    std::cout << "========================================================\n";
    std::cout << "   Decision Tree: Normal vs Chunks (Localidade Temporal)\n";
    std::cout << "   Dataset: Forest Cover Type\n";
    std::cout << "========================================================\n\n";
    
    // Carregar dataset
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    
    // Usar subconjunto para testes (ajuste conforme necessário)
    const int max_samples = 100000;  // 100k amostras para teste rápido
    
    std::cout << "Carregando dataset (max: " << max_samples << " amostras)...\n";
    
    try {
        DataLoader::load_csv("covertype_dataset.csv", X, y, max_samples);
        std::cout << "\nDataset carregado: " << X.size() << " amostras, " 
                  << X[0].size() << " features\n\n";
    } catch (const std::exception& e) {
        std::cerr << "❌ Erro ao carregar dataset: " << e.what() << "\n";
        std::cerr << "Execute primeiro: python download_covertype.py\n";
        return 1;
    }
    
    // Número de execuções para média (reduzir para datasets grandes)
    const int num_runs = 3;
    
    std::cout << "Executando " << num_runs << " iteracoes para obter medias...\n";
    std::cout << std::string(63, '-') << "\n";
    
    double total_time_normal = 0.0;
    double total_time_chunks = 0.0;
    
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "\nIteracao " << (run + 1) << "/" << num_runs << ":\n";
        
        // ========================================
        // Decision Tree NORMAL (acesso aleatório)
        // ========================================
        std::cout << "  Treinando versao Normal... ";
        DecisionTree tree_normal(8, 5);
        auto start = std::chrono::high_resolution_clock::now();
        tree_normal.fit(X, y, false);  // false = sem chunks
        auto end = std::chrono::high_resolution_clock::now();
        double time_normal = std::chrono::duration<double, std::milli>(end - start).count();
        total_time_normal += time_normal;
        std::cout << time_normal << " ms\n";
        
        // ========================================
        // Decision Tree COM CHUNKS (localidade temporal)
        // ========================================
        std::cout << "  Treinando versao com Chunks... ";
        DecisionTree tree_chunks(8, 5);
        start = std::chrono::high_resolution_clock::now();
        tree_chunks.fit(X, y, true);  // true = com chunks de 100
        end = std::chrono::high_resolution_clock::now();
        double time_chunks = std::chrono::duration<double, std::milli>(end - start).count();
        total_time_chunks += time_chunks;
        std::cout << time_chunks << " ms\n";
    }
    
    // Calcular médias
    double avg_time_normal = total_time_normal / num_runs;
    double avg_time_chunks = total_time_chunks / num_runs;
    double speedup = avg_time_normal / avg_time_chunks;
    
    // Exibir resultados
    std::cout << std::setw(20) << "Método"
              << std::setw(20) << "Tempo Médio (ms)"
              << std::setw(15) << "Speedup\n";
    std::cout << std::string(55, '-') << "\n";
    
    std::cout << std::setw(20) << "Normal"
              << std::setw(20) << std::fixed << std::setprecision(4) 
              << avg_time_normal
              << std::setw(15) << "-\n";
    
    std::cout << std::setw(20) << "Chunks (100)"
              << std::setw(20) << std::fixed << std::setprecision(4) 
              << avg_time_chunks
              << std::setw(14) << std::setprecision(2) 
              << speedup << "x\n";
    
    // Salvar em CSV
    std::ofstream csv("results.csv");
    csv << "Metodo,TempoMedio(ms),Speedup\n";
    csv << "Normal," << avg_time_normal << ",1.00\n";
    csv << "Chunks," << avg_time_chunks << "," << speedup << "\n";
    csv.close();
    
    std::cout << "\n========================================================\n";
    std::cout << "Resultados salvos em: results.csv\n";
    std::cout << "========================================================\n";
    return 0;
}
