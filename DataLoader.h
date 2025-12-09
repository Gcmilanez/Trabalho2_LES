#ifndef DATALOADER_H
#define DATALOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept>

class DataLoader {
public:
    static void load_csv(const std::string& path, 
                         std::vector<std::vector<double>>& X, 
                         std::vector<int>& y, 
                         int max_samples = -1) 
    {
        std::ifstream file(path);
        if (!file.is_open()) {
            throw std::runtime_error("Nao foi possivel abrir o arquivo: " + path);
        }

        std::string line;
        // Tenta ler o cabeçalho (opcional, remova se seus CSVs não tiverem header)
        // Se a primeira linha já for dados e der erro de conversão, comente a linha abaixo.
        if(std::getline(file, line)) {
            // Verifica se é header (contem letras) ou dados
            // Lógica simples: se tentar converter e falhar, é header.
            try {
                std::stringstream ss(line);
                std::string cell;
                if(std::getline(ss, cell, ',')) {
                    std::stod(cell); 
                    // Se converteu, volta o ponteiro do arquivo para o início
                    file.clear();
                    file.seekg(0);
                }
            } catch (...) {
                // É header, ignora e segue
            }
        }

        X.clear();
        y.clear();

        while (std::getline(file, line)) {
            if (max_samples > 0 && (int)X.size() >= max_samples) break;
            if (line.empty()) continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            
            while (std::getline(ss, cell, ',')) {
                try {
                    row.push_back(std::stod(cell));
                } catch (...) {
                    // Ignora células mal formadas ou vazias
                }
            }
            
            // Assume que a última coluna é o Target (y)
            if (!row.empty()) {
                y.push_back((int)row.back());
                row.pop_back();
                X.push_back(row);
            }
        }
    }
};

#endif