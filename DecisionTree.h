#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include <vector>
#include <memory>

struct Node {
    bool is_leaf = false;
    int feature_index = -1;
    double threshold = 0.0;
    int predicted_class = -1;
    std::unique_ptr<Node> left;
    std::unique_ptr<Node> right;
};

class DecisionTree {
private:
    int max_depth;
    int min_samples_split;
    std::unique_ptr<Node> root;
    
    // Flag para usar processamento em chunks
    bool use_chunk_processing;
    static const int CHUNK_SIZE = 100;
    
    // Buffers pré-alocados para versão com chunks
    std::vector<double> chunk_buffer;
    std::vector<int> chunk_labels_buffer;
    
    // Contadores para análise de performance
    mutable size_t cache_friendly_accesses;
    mutable size_t random_accesses;
    
    // Métodos auxiliares
    double calculate_gini(const std::vector<int>& labels) const;
    
    struct SplitResult {
        double threshold;
        double gain;
        std::vector<int> left_indices;
        std::vector<int> right_indices;
    };
    
    // ============================================================
    // VERSÃO GENÉRICA (BÁSICA) - Acesso aleatório à memória
    // ============================================================
    // Problema: Cada acesso pode causar cache miss
    // - Dados dispersos na memória
    // - CPU precisa buscar da RAM frequentemente
    // - Latência: ~100-200 ciclos por cache miss
    SplitResult find_best_split_basic(const std::vector<int>& indices,
                                      int feature_index,
                                      const std::vector<std::vector<double>>& X,
                                      const std::vector<int>& y);
    
    // ============================================================
    // VERSÃO OTIMIZADA - Chunks de 100 elementos
    // ============================================================
    // Processa dados em blocos de 100:
    // 1. Carrega 100 elementos (800 bytes no cache L1)
    // 2. Processa IMEDIATAMENTE enquanto estão "quentes"
    // 3. Repete para próximo chunk
    // 
    // LOCALIDADE TEMPORAL: Dados são usados rapidamente após
    // carregamento, permanecendo no cache durante processamento
    // 
    // Benefício: Cache hit rate 95%+ vs 20-30% (versão normal)
    SplitResult find_best_split_chunked(const std::vector<int>& indices,
                                        int feature_index,
                                        const std::vector<std::vector<double>>& X,
                                        const std::vector<int>& y);
    
    std::unique_ptr<Node> build_tree(const std::vector<int>& indices,
                                     const std::vector<std::vector<double>>& X,
                                     const std::vector<int>& y,
                                     int depth);
    
    int predict_sample(const std::vector<double>& sample, const Node* node) const;
    
public:
    DecisionTree(int max_depth = 5, int min_samples_split = 2);
    
    // Treinar modelo
    // use_chunks = false: Versão GENÉRICA (acesso aleatório)
    // use_chunks = true:  Versão OTIMIZADA (chunks de 100 para cache locality)
    void fit(const std::vector<std::vector<double>>& X, 
             const std::vector<int>& y,
             bool use_chunks = false);
    
    std::vector<int> predict(const std::vector<std::vector<double>>& X) const;
    
    // Estatísticas de acesso à memória
    size_t get_cache_friendly_accesses() const { return cache_friendly_accesses; }
    size_t get_random_accesses() const { return random_accesses; }
    void reset_access_counters() { cache_friendly_accesses = 0; random_accesses = 0; }
};

#endif
