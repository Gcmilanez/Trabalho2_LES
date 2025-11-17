Intregrantes:

Giovanni Milanez
Gabriel Welter

Objetivo do trabalho:
• Otimizar um algoritmo utilizando cache

O que definimos:

• Algoritmo escolhido foi do nosso grupo foi Decision Tree
• Montar um gráfico comparando o benchmark do algoritmo genérico x algoritmo otimizado

Linguagens utilizadas:

• C++

Ideias base para implementação:

• Utilizar chunks de 100 blocos que garante a localidade espacial

## Estrutura do Projeto

```
DecisionTree.h       - Declarações das classes e estruturas
DecisionTree.cpp     - Implementação do algoritmo
benchmark.cpp        - Sistema de benchmark e comparação
```

## Comparação

### Versão BÁSICA
- Acesso aleatório aos dados
- Cache misses frequentes
- Sem otimização de localidade

### Versão OTIMIZADA (Com Chunks)
- Processamento em blocos de 100 elementos
- Melhor localidade espacial (dados contíguos no cache L1/L2)
- Melhor localidade temporal (dados processados enquanto estão no cache)
- Redução de cache misses

## Como Funciona a Otimização

A versão otimizada processa os dados em **blocos de 100 elementos**:

1. **Carrega um chunk** (100 elementos) na memória cache
2. **Processa todo o chunk** enquanto os dados estão "quentes" no cache
3. **Passa para o próximo chunk**

Isso garante que:
- Os dados acessados estão próximos na memória (localidade espacial)
- Os dados são reutilizados rapidamente (localidade temporal)
- Minimiza cache misses e aumenta performance

## ATENÇÃO
Dataset é ficticio, usar um dataset real para resultados mais precisos

## Compilação

```bash
g++ -std=c++17 -O3 -o benchmark benchmark.cpp DecisionTree.cpp
./benchmark
```

## Execução

```bash
./benchmark
```

O programa irá:
- Executar benchmarks com diferentes tamanhos de dataset (100 a 5000 amostras)
- Testar com diferentes quantidades de features (5, 10, 15)
- Comparar versão básica vs versão com chunks
- Gerar arquivo `benchmark_results.csv` com os resultados

## Resultados Esperados

O arquivo CSV conterá:
- Tamanho do dataset
- Número de features
- Tempo de execução versão básica
- Tempo de execução versão com chunks
- Speedup obtido

Use esses dados para gerar gráficos de comparação de performance.

## Observações

O speedup será mais evidente com datasets maiores, pois:
- Mais dados = mais acessos à memória
- Chunks otimizam o uso do cache de CPU
- Reduzem latência de acesso à memória RAM