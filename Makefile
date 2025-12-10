# ============================================================
#   Makefile - Random Forest (Unificada)
#   - Treino + Salvamento (Baseline & Optimized)
#   - Load + Predicao (Baseline & Optimized)
# ============================================================

CXX      := g++
CXXFLAGS := -std=c++17 -O3 -Wall -Wextra -march=native
LDFLAGS  := 

OBJ_DIR  := obj

# Cria o diretório obj se não existir
$(shell mkdir -p $(OBJ_DIR))

# ------------------------------------------------------------
# 1) Executavel - Treino Baseline
# ------------------------------------------------------------

FOREST_BASELINE_TRAIN_OBJS := \
	$(OBJ_DIR)/DecisionTree.o \
	$(OBJ_DIR)/RandomForest.o \
	$(OBJ_DIR)/main_forest_baseline.o

forest_baseline_train: $(FOREST_BASELINE_TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✔ Executavel gerado: ./forest_baseline_train"

# ------------------------------------------------------------
# 2) Executavel - Treino Otimizado
# ------------------------------------------------------------

FOREST_OPTIMIZED_TRAIN_OBJS := \
	$(OBJ_DIR)/DecisionTree.o \
	$(OBJ_DIR)/RandomForest.o \
	$(OBJ_DIR)/main_forest_optimized.o

forest_optimized_train: $(FOREST_OPTIMIZED_TRAIN_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✔ Executavel gerado: ./forest_optimized_train"

# ------------------------------------------------------------
# 3) Executavel - Predicao Baseline
# ------------------------------------------------------------

FOREST_BASELINE_PREDICT_OBJS := \
	$(OBJ_DIR)/DecisionTree.o \
	$(OBJ_DIR)/RandomForest.o \
	$(OBJ_DIR)/main_predict_baseline.o

forest_baseline_predict: $(FOREST_BASELINE_PREDICT_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✔ Executavel gerado: ./forest_baseline_predict"

# ------------------------------------------------------------
# 4) Executavel - Predicao Otimizada
# ------------------------------------------------------------

FOREST_OPTIMIZED_PREDICT_OBJS := \
	$(OBJ_DIR)/DecisionTree.o \
	$(OBJ_DIR)/RandomForest.o \
	$(OBJ_DIR)/main_predict_optimized.o

forest_optimized_predict: $(FOREST_OPTIMIZED_PREDICT_OBJS)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✔ Executavel gerado: ./forest_optimized_predict"

# ------------------------------------------------------------
# Regras de compilacao dos .cpp -> obj/
# ------------------------------------------------------------

# Compila a Arvore de Decisão
$(OBJ_DIR)/DecisionTree.o: DecisionTree.cpp DecisionTree.h
	$(CXX) $(CXXFLAGS) -c DecisionTree.cpp -o $@

# Compila a Random Forest (Classe Unificada)
$(OBJ_DIR)/RandomForest.o: RandomForest.cpp RandomForest.h DecisionTree.h
	$(CXX) $(CXXFLAGS) -c RandomForest.cpp -o $@

# Compila os Mains
# Note que agora todos dependem de RandomForest.h (unificado) e DataLoader.h

$(OBJ_DIR)/main_forest_baseline.o: main_forest_baseline.cpp RandomForest.h DataLoader.h
	$(CXX) $(CXXFLAGS) -c main_forest_baseline.cpp -o $@

$(OBJ_DIR)/main_forest_optimized.o: main_forest_optimized.cpp RandomForest.h DataLoader.h
	$(CXX) $(CXXFLAGS) -c main_forest_optimized.cpp -o $@

$(OBJ_DIR)/main_predict_baseline.o: main_predict_baseline.cpp RandomForest.h DataLoader.h
	$(CXX) $(CXXFLAGS) -c main_predict_baseline.cpp -o $@

$(OBJ_DIR)/main_predict_optimized.o: main_predict_optimized.cpp RandomForest.h DataLoader.h
	$(CXX) $(CXXFLAGS) -c main_predict_optimized.cpp -o $@

# ------------------------------------------------------------
# Alvo padrao: compilar tudo
# ------------------------------------------------------------

all: forest_baseline_train forest_optimized_train \
     forest_baseline_predict forest_optimized_predict
	@echo "============================================================"
	@echo " Executaveis compilados com sucesso!"
	@echo "  → ./forest_baseline_train"
	@echo "  → ./forest_optimized_train"
	@echo "  → ./forest_baseline_predict"
	@echo "  → ./forest_optimized_predict"
	@echo "============================================================"

# ------------------------------------------------------------
# Limpeza
# ------------------------------------------------------------

clean:
	rm -rf $(OBJ_DIR)/*.o \
		forest_baseline_train forest_optimized_train \
		forest_baseline_predict forest_optimized_predict
	@echo "✔ Arquivos de compilacao removidos."

.PHONY: all clean