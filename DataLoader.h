#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <iostream>

class DataLoader {
public:
    // Carrega CSV com número automático de features
    // Última coluna = label, demais = features
    static void load_csv(const std::string& filename,
                        std::vector<std::vector<double>>& X,
                        std::vector<int>& y,
                        int max_samples = -1) {
        std::ifstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Não foi possível abrir o arquivo: " + filename);
        }
        
        std::string line;
        // Pular header
        std::getline(file, line);
        
        X.clear();
        y.clear();
        
        int samples_loaded = 0;
        
        while (std::getline(file, line)) {
            if (max_samples > 0 && samples_loaded >= max_samples) {
                break;
            }
            
            std::stringstream ss(line);
            std::string value;
            std::vector<double> features;
            std::vector<std::string> tokens;
            
            // Ler todos os valores
            while (std::getline(ss, value, ',')) {
                tokens.push_back(value);
            }
            
            if (tokens.empty()) continue;
            
            // Todas as colunas exceto última são features
            for (size_t i = 0; i < tokens.size() - 1; ++i) {
                features.push_back(std::stod(tokens[i]));
            }
            
            // Última coluna é o label
            int label = std::stoi(tokens.back());
            
            X.push_back(features);
            y.push_back(label);
            samples_loaded++;
            
            // Progresso a cada 100k amostras
            if (samples_loaded % 100000 == 0) {
                std::cout << "  Carregadas " << samples_loaded << " amostras...\n";
            }
        }
        
        file.close();
    }
};

#endif
