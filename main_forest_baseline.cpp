#include "RandomForestBaseline.h"
#include "DataLoader.h"

#include <iostream>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <string>

// helper simples para extrair o nome do arquivo a partir do caminho
std::string get_filename_only(const std::string& path) {
    std::size_t pos = path.find_last_of("/\\");
    if (pos == std::string::npos) return path;
    return path.substr(pos + 1);
}

int main(int argc, char** argv) {
    std::cout << "========================================================\n";
    std::cout << "   Random Forest (Baseline): Implementacao Comum\n";
    std::cout << "========================================================\n\n";

    if (argc < 2) {
        std::cerr << "Uso: " << argv[0]
                  << " <arquivo_dataset.csv> [max_samples] [num_runs]\n";
        std::cerr << "Exemplo: " << argv[0]
                  << " covertype_dataset.csv 100000 3\n";
        return 1;
    }

    std::string dataset_path = argv[1];

    int max_samples = 100000; // padrão
    if (argc >= 3) {
        max_samples = std::stoi(argv[2]);
    }

    int num_runs = 3; // padrão
    if (argc >= 4) {
        num_runs = std::stoi(argv[3]);
    }

    std::cout << "Dataset: " << dataset_path << "\n";
    std::cout << "Max samples: " << max_samples << "\n";
    std::cout << "Num runs: " << num_runs << "\n\n";

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
        std::cout << "\nDataset carregado: " << X.size() << " amostras, "
                  << X[0].size() << " features\n\n";
    }
    catch (const std::exception& e) {
        std::cerr << "❌ Erro ao carregar dataset: " << e.what() << "\n";
        return 1;
    }

    // Hiperparametros (fixos para comparacao justa)
    const int n_trees           = 50;
    const int max_depth         = 8;
    const int min_samples_split = 5;

    std::cout << "Executando " << num_runs
              << " iteracoes para obter medias...\n";
    std::cout << std::string(63, '-') << "\n";

    double total_time_ms = 0.0;

    for (int run = 0; run < num_runs; ++run) {
        std::cout << "\nIteracao " << (run + 1) << "/" << num_runs << ":\n";

        std::cout << "  Treinando RandomForestBaseline... ";
        RandomForestBaseline forest(n_trees, max_depth, min_samples_split);

        auto start = std::chrono::high_resolution_clock::now();
        forest.fit(X, y);
        auto end = std::chrono::high_resolution_clock::now();

        double time_ms =
            std::chrono::duration<double, std::milli>(end - start).count();
        total_time_ms += time_ms;
        std::cout << time_ms << " ms\n";
    }

    double avg_time_ms = total_time_ms / num_runs;

    std::cout << "\n";
    std::cout << std::setw(25) << "Metodo"
              << std::setw(20) << "Tempo Medio (ms)\n";
    std::cout << std::string(45, '-') << "\n";

    std::cout << std::setw(25) << "RandomForest Baseline"
              << std::setw(20) << std::fixed << std::setprecision(4)
              << avg_time_ms << "\n";

    // Nome do CSV inclui nome do dataset
    std::string base_name = get_filename_only(dataset_path);
    std::string csv_name = "results_forest_baseline_" + base_name + ".csv";

    std::ofstream csv(csv_name);
    csv << "Metodo,Dataset,MaxSamples,NumRuns,TempoMedio(ms)\n";
    csv << "RandomForestBaseline,"
        << base_name << ","
        << max_samples << ","
        << num_runs << ","
        << avg_time_ms << "\n";
    csv.close();

    std::cout << "\n========================================================\n";
    std::cout << "Resultados salvos em: " << csv_name << "\n";
    std::cout << "========================================================\n";
    return 0;
}
