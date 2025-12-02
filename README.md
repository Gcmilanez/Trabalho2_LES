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

