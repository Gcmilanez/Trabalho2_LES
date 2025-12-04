# ============================================================
#   Makefile para compilar:
#      - Random Forest Baseline
#      - Random Forest Otimizada
#
#   O main original (main.cpp) foi removido da build.
# ============================================================

CXX       := g++
CXXFLAGS  := -std=c++17 -O3 -Wall -Wextra -march=native
LDFLAGS   := 

SRC_DIR   := .
OBJ_DIR   := obj

$(shell mkdir -p $(OBJ_DIR))

# ------------------------------------------------------------
# Fontes comuns às duas florestas
# DecisionTree.cpp é compartilhado
# DataLoader é header-only e NÃO entra aqui
# ------------------------------------------------------------
COMMON_SRC := \
    $(SRC_DIR)/DecisionTree.cpp

# ------------------------------------------------------------
# 1) Executável – Random Forest Baseline
# ------------------------------------------------------------
FOREST_BASELINE_SRC := \
    $(COMMON_SRC) \
    $(SRC_DIR)/RandomForestBaseline.cpp \
    $(SRC_DIR)/main_forest_baseline.cpp

FOREST_BASELINE_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FOREST_BASELINE_SRC))

forest_baseline: $(FOREST_BASELINE_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✔ Executável gerado: ./forest_baseline"

# ------------------------------------------------------------
# 2) Executável – Random Forest Otimizada
# ------------------------------------------------------------
FOREST_OPTIMIZED_SRC := \
    $(COMMON_SRC) \
    $(SRC_DIR)/RandomForestOptimized.cpp \
    $(SRC_DIR)/main_forest_optimized.cpp

FOREST_OPTIMIZED_OBJ := $(patsubst $(SRC_DIR)/%.cpp,$(OBJ_DIR)/%.o,$(FOREST_OPTIMIZED_SRC))

forest_optimized: $(FOREST_OPTIMIZED_OBJ)
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)
	@echo "✔ Executável gerado: ./forest_optimized"

# ------------------------------------------------------------
# Regra geral de compilação
# ------------------------------------------------------------
$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# ------------------------------------------------------------
# "make all"
# ------------------------------------------------------------
all: forest_baseline forest_optimized
	@echo "============================================================"
	@echo " Executáveis compilados com sucesso!"
	@echo "  → ./forest_baseline"
	@echo "  → ./forest_optimized"
	@echo "============================================================"

# ------------------------------------------------------------
# Limpeza
# ------------------------------------------------------------
clean:
	rm -rf $(OBJ_DIR)/*.o forest_baseline forest_optimized
	@echo "✔ Arquivos de compilação removidos."

.PHONY: all clean
