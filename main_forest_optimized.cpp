#include "RandomForestOptimized.h"
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
    std::cout << "   Random Forest Otimizada: TREINO + SALVAMENTO\n";
    std::cout << "========================================================\n\n";

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0]
                  << " <arquivo_dataset.csv> [max_samples] [num_runs] [modelo_saida]\n";
        std::cerr << "Exemplo: " << argv[0]
                  << " covertype_dataset.csv 100000 1 optimized.model\n";
        return 1;
    }

    std::string dataset_path = argv[1];

    int max_samples = 100000;
    if (argc >= 3) {
        max_samples = std::stoi(argv[2]);
    }

    int num_runs = 1;
    if (argc >= 4) {
        num_runs = std::stoi(argv[3]);
    }

    std::string model_path;
    if (argc >= 5) {
        model_path = argv[4];
    } else {
        model_path = "optimized_" + get_filename_only(dataset_path) + ".model";
    }

    std::cout << "Dataset     : " << dataset_path << "\n";
    std::cout << "Max samples : " << max_samples << "\n";
    std::cout << "Num runs    : " << num_runs << "\n";
    std::cout << "Modelo saida: " << model_path << "\n\n";

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

    // Hiperparâmetros (ajuste conforme seu projeto)
    const int n_trees           = 32;
    const int max_depth         = 8;
    const int min_samples_split = 5;
    const int chunk_size        = 100;

    double total_train_ms = 0.0;

    RandomForestOptimized forest(n_trees, max_depth,
                                 min_samples_split, chunk_size);

    for (int run = 0; run < num_runs; ++run) {
        std::cout << "Iteracao " << (run + 1) << "/" << num_runs << "...\n";

        auto start_train = std::chrono::high_resolution_clock::now();
        forest.fit(X, y);
        auto end_train   = std::chrono::high_resolution_clock::now();

        double train_ms =
            std::chrono::duration<double, std::milli>(end_train - start_train).count();
        total_train_ms += train_ms;

        std::cout << "  Tempo treino: " << train_ms << " ms\n";
    }

    double avg_train_ms = total_train_ms / num_runs;

    std::cout << "\nSalvando modelo em: " << model_path << "\n";
    forest.save_model(model_path);

    std::cout << "\n================= RESULTADOS TREINO =====================\n";
    std::cout << std::setw(25) << "Tempo Treino Medio (ms)"
              << std::setw(20) << std::fixed << std::setprecision(4)
              << avg_train_ms << "\n";
    std::cout << "========================================================\n";

    std::string csv_name = "results_forest_optimized_train_" +
                           get_filename_only(dataset_path) + ".csv";
    std::ofstream csv(csv_name);
    csv << "Metodo,Dataset,MaxSamples,NumRuns,TempoTreinoMedio(ms),Modelo\n";
    csv << "RandomForestOptimizedTrain,"
        << get_filename_only(dataset_path) << ","
        << max_samples << ","
        << num_runs << ","
        << avg_train_ms << ","
        << model_path << "\n";
    csv.close();

    std::cout << "Resultados salvos em: " << csv_name << "\n";
    return 0;
}
