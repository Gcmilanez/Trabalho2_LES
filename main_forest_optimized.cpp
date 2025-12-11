#include "RandomForest.h"
#include "DataLoader.h"
#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>

std::string get_filename_only(const std::string& path) {
    std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "   RF Otimizada: TREINO (FLAT MEMORY)\n";
    std::cout << "========================================================\n\n";

    // Verificação de argumentos rígida para evitar erro de stoi
    // Esperado: exe <dataset> <max_samples> <num_runs> <model_out>
    if (argc < 5) {
        std::cerr << "Uso incorreto!\n";
        std::cerr << "Uso: " << argv[0] << " <dataset.csv> <max_samples> <num_runs> <model_out>\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    // O erro de stoi acontecia aqui se a ordem estivesse trocada (ex: lendo nome do arquivo como numero)
    int max_samples = std::stoi(argv[2]);
    int num_runs    = std::stoi(argv[3]);
    std::string model_path = argv[4];

    std::cout << "Dataset: " << dataset_path << "\n";
    std::cout << "Samples: " << max_samples << "\n";
    std::cout << "Runs   : " << num_runs << "\n";
    std::cout << "Output : " << model_path << "\n\n";

    // 1. Carregar Dataset
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    try {
        DataLoader::load_csv(dataset_path, X, y, max_samples);
        if (X.empty()) throw std::runtime_error("Dataset vazio");
    } catch (const std::exception& e) {
        std::cerr << "Erro load: " << e.what() << "\n";
        return 1;
    }

    std::cout << "Dados carregados. Iniciando conversao para Flat Memory...\n";

    // =================================================================
    // 2. CONVERSÃO PARA MEMÓRIA LINEAR (FLAT) - ANTES DO TEMPO
    // =================================================================
    // Isso garante que o tempo de treino seja justo, medindo apenas o algoritmo
    // e não a alocação de memória auxiliar.
    
    // NOTA: fit_optimized dentro da classe RandomForest espera vector<vector>
    // e faz a conversão interna, OU podemos adaptar para receber flat.
    // Para manter compatibilidade com sua classe atual que aceita vector<vector> no fit_optimized,
    // vamos passar o X normal, mas a RandomForest::fit_optimized JÁ FAZ a conversão interna 
    // eficiente baseada em blocos.
    
    // Se você quiser EXTREMA performance, o ideal seria passar o flat direto, 
    // mas sua implementação atual de fit_optimized (Source 2) já trata isso criando o 'X_flat_global'.
    // Então, chamaremos fit_optimized normalmente.
    
    // Configuração da Floresta
    const int N_TREES = 50;
    const int MAX_DEPTH = 8;
    const int MIN_SAMPLES_SPLIT = 5;
    const int CHUNK_SIZE = 64; 

    RandomForest forest(N_TREES, MAX_DEPTH, MIN_SAMPLES_SPLIT, CHUNK_SIZE);

    double total_ms = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << num_runs << "... ";
        
        auto start = std::chrono::high_resolution_clock::now();
        
        // Chamada OTIMIZADA
        // Internamente ela cria o layout de memória flat e treina
        forest.fit_optimized(X, y); 
        
        auto end = std::chrono::high_resolution_clock::now();
        
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        std::cout << ms << " ms\n";
    }

    std::cout << "\nSalvando modelo em: " << model_path << "\n";
    forest.save_model(model_path);

    double avg_ms = total_ms / num_runs;
    std::cout << "Media Otimizada: " << avg_ms << " ms\n";

    // CSV Output
    std::string csv_name = "results_forest_optimized_train_" + get_filename_only(dataset_path) + ".csv";
    std::ofstream csv(csv_name);
    csv << "Metodo,Dataset,MaxSamples,Runs,TempoMedio(ms)\n";
    csv << "Optimized," << get_filename_only(dataset_path) << "," << max_samples << "," << num_runs << "," << avg_ms << "\n";
    
    return 0;
}