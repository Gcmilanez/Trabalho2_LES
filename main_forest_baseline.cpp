#include "RandomForest.h" // Classe unificada
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
    std::cout << "   Random Forest (Baseline): TREINO\n";
    std::cout << "   [Parametros da Imagem: 50 arvores, Depth 8, Split 5]\n";
    std::cout << "========================================================\n\n";

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0] << " <dataset.csv> [max_samples] [num_runs] [model_out]\n";
        return 1;
    }

    std::string dataset_path = argv[1];
    int max_samples = (argc >= 3) ? std::stoi(argv[2]) : 100000;
    int num_runs = (argc >= 4) ? std::stoi(argv[3]) : 1;
    std::string model_path = (argc >= 5) ? argv[4] : "models/baseline_model.bin";

    // 1. Carregar Dataset
    std::vector<std::vector<double>> X;
    std::vector<int> y;
    try {
        DataLoader::load_csv(dataset_path, X, y, max_samples);
        if (X.empty()) throw std::runtime_error("Dataset vazio");
        std::cout << "Dataset: " << X.size() << " amostras carregadas.\n";
    } catch (const std::exception& e) {
        std::cerr << "Erro: " << e.what() << "\n";
        return 1;
    }

    // 2. Configurar Hiperparâmetros (Conforme sua imagem)
    const int N_TREES = 50;
    const int MAX_DEPTH = 8;
    const int MIN_SAMPLES_SPLIT = 5;
    
    // Instancia a classe unificada
    RandomForest forest(N_TREES, MAX_DEPTH, MIN_SAMPLES_SPLIT);

    double total_ms = 0.0;

    // 3. Loop de Treino
    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Run " << (run + 1) << "/" << num_runs << "... ";
        auto start = std::chrono::high_resolution_clock::now();
        
        // --- AQUI ESTÁ A DIFERENÇA ---
        // false = Modo Baseline (Lento, Row-Major, O(N^2))
        forest.fit(X, y, false); 
        
        auto end = std::chrono::high_resolution_clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        total_ms += ms;
        std::cout << ms << " ms\n";
    }

    // 4. Salvar e Relatar
    std::cout << "\nSalvando modelo em: " << model_path << "\n";
    forest.save_model(model_path); // Certifique-se que RandomForest tem save_model implementado

    double avg_ms = total_ms / num_runs;
    std::cout << "Media Baseline: " << avg_ms << " ms\n";

    // CSV para logs
    std::string csv_name = "results_forest_baseline_train_" + get_filename_only(dataset_path) + ".csv";
    std::ofstream csv(csv_name);
    csv << "Metodo,Dataset,MaxSamples,Runs,TempoMedio(ms)\n";
    csv << "Baseline," << get_filename_only(dataset_path) << "," << max_samples << "," << num_runs << "," << avg_ms << "\n";
    
    return 0;
}