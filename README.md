## Intregrantes:

Giovanni Milanez
Gabriel Welter

Objetivo do trabalho:
• Otimizar um algoritmo utilizando cache

O que definimos:

• Algoritmo escolhido: Decision Tree
• Comparação: Decision Tree Normal vs Decision Tree com Chunks de 100
• Foco: Aproveitar localidade temporal do cache

Linguagens utilizadas:

• C++
• Python

## Estrutura do Projeto

```
DecisionTree.h    - Interface da Decision Tree
DecisionTree.cpp  - Implementação (normal + otimizada com chunks)
main.cpp          - Programa principal de comparação
```

## Como Funciona a Otimização

Processamento em **blocos de 100 elementos**:

1. **Carrega um chunk** (100 elementos = 800 bytes)
2. **Processa todo o chunk** enquanto os dados estão "quentes" no cache
3. **Passa para o próximo chunk**

## Compilação e Execução

```bash
g++ -std=c++17 -O3 -o programa.exe main.cpp DecisionTree.cpp
.\programa.exe
```

## Resultados

O programa gera:
- Saída no console com tempos de execução
- Arquivo `results.csv` com dados para gráficos
- Speedup para cada tamanho de dataset

📂 Estrutura do Projeto
🔵 1. DecisionTree.h / DecisionTree.cpp

Implementa uma árvore de decisão binária com:

Cálculo do índice Gini

Critérios de parada configuráveis

Divisão básica ou usando chunks

Contadores de acesso à memória:

cache_friendly_accesses

random_accesses

Métodos de predição:

predict_one

predict

Serialização binária:

save_model(std::ostream&)

load_model(std::istream&)

Suporte a move semantics (necessário para std::vector<DecisionTree>)

Esta é a base para ambas as Random Forests.

🔵 2. RandomForestBaseline.h / RandomForestBaseline.cpp

Implementação tradicional de Random Forest:

Amostragem bootstrap com reposição

Árvores independentes

Votação majoritária

Serialização binária da floresta:

save_model(filename)

load_model(filename)

Serve como referência comparativa para o modelo otimizado.

🔵 3. RandomForestOptimized.h / RandomForestOptimized.cpp

Versão otimizada para melhorar a eficiência energética, com:

Reorganização cache-friendly dos índices para cada árvore

Uso obrigatório do modo chunked na DecisionTree

Mesmo formato de serialização

Predição idêntica à baseline, porém com estrutura interna mais eficiente

É a versão destinada ao experimento principal.

🔵 4. DataLoader.h

Carrega datasets CSV no formato:

f1, f2, f3, ..., fN, classe


Lê:

X → matriz de atributos

y → vetor de classes

max_samples → limite opcional de leitura

🔵 5. Arquivos main_*

O projeto contém quatro programas principais, cada um com uma função clara:

✔ main_forest_baseline.cpp

Treina o modelo baseline e salva em arquivo.

✔ main_forest_optimized.cpp

Treina o modelo otimizado e salva em arquivo.

✔ main_predict_baseline.cpp

Carrega modelo baseline e executa predição isolada.

✔ main_predict_optimized.cpp

Carrega modelo otimizado e executa predição isolada.

Esses quatro programas permitem medir treino e predição independentemente, o que é essencial para experimentos com métricas energéticas.

⚙️ Compilação

Basta rodar:

make clean
make -j


Serão gerados os executáveis:

forest_baseline_train
forest_optimized_train
forest_baseline_predict
forest_optimized_predict

🚀 Como Utilizar
1. Treinar modelo baseline
./forest_baseline_train dataset.csv 100000 1 baseline.model


Argumentos:

arquivo CSV

quantidade máxima de amostras

número de execuções (geralmente 1)

arquivo de saída do modelo

2. Treinar modelo otimizado
./forest_optimized_train dataset.csv 100000 1 optimized.model

3. Predição com o modelo baseline
./forest_baseline_predict dataset.csv baseline.model 100000 3


Argumentos:

dataset de teste

arquivo .model

número de amostras

número de execuções de predição

4. Predição com o modelo otimizado
./forest_optimized_predict dataset.csv optimized.model 100000 3

⚡ Medição Energética com perf
Medir treino da versão otimizada:
perf stat -e power/energy-cores/,power/energy-pkg/ \
    ./forest_optimized_train dataset.csv 100000 1 optimized.model

Medir predição da versão otimizada:
perf stat -e power/energy-cores/,power/energy-pkg/ \
    ./forest_optimized_predict dataset.csv optimized.model 100000 3


Eventos úteis:

cycles

instructions

LLC-load-misses

power/energy-cores/

power/energy-pkg/

📈 Resultados Esperados

A Random Forest Optimized deve apresentar:

menos random accesses

mais acessos sequenciais (cache-friendly)

menor consumo energético em predição

ligeiro aumento de custo de treino (dependendo do dataset)

mesma exatidão (mesmo algoritmo básico)

A versão Baseline atua como controle experimental.

🧠 Considerações Importantes

As árvores são completamente serializáveis, permitindo testes isolados.

A lógica de chunks reduz pressão na cache durante splits.

O uso de construtores de movimento impede operações caras de cópia.

Todas os executáveis foram projetados para funcionar com datasets arbitrários.