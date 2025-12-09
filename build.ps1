# ============================================================
#   Script de Compilação Unificado (Classes Otimizadas)
# ============================================================

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Compilando Random Forest (Unified Architecture)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Criar diretório obj
if (-not (Test-Path "obj")) { New-Item -ItemType Directory -Path "obj" | Out-Null }

$CXX = "g++"
# Flags agressivas para benchmark real
$CXXFLAGS = "-std=c++17 -O3 -Wall -Wextra -march=native" 

# ------------------------------------------------------------
# 1. Compilar Classes Base (Objetos)
# ------------------------------------------------------------

Write-Host "[1/6] Compilando DecisionTree.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c DecisionTree.cpp -o obj/DecisionTree.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[2/6] Compilando RandomForest.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c RandomForest.cpp -o obj/RandomForest.o
if ($LASTEXITCODE -ne 0) { exit 1 }

# ------------------------------------------------------------
# 2. Compilar Mains (Treino)
# ------------------------------------------------------------

Write-Host "[3/6] Compilando main_forest_baseline.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_forest_baseline.cpp -o obj/main_forest_baseline.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[4/6] Compilando main_forest_optimized.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_forest_optimized.cpp -o obj/main_forest_optimized.o
if ($LASTEXITCODE -ne 0) { exit 1 }

# ------------------------------------------------------------
# 3. Compilar Mains (Predicao)
# ------------------------------------------------------------

Write-Host "[5/6] Compilando main_predict_baseline.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_predict_baseline.cpp -o obj/main_predict_baseline.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[6/6] Compilando main_predict_optimized.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_predict_optimized.cpp -o obj/main_predict_optimized.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "Linkando executaveis..." -ForegroundColor Green

# ------------------------------------------------------------
# 4. Linkagem Final
# ------------------------------------------------------------

# Lista comum de objetos das classes
$coreObjs = "obj/DecisionTree.o", "obj/RandomForest.o"

Write-Host "  -> forest_baseline_train.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() $coreObjs "obj/main_forest_baseline.o" -o forest_baseline_train.exe

Write-Host "  -> forest_optimized_train.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() $coreObjs "obj/main_forest_optimized.o" -o forest_optimized_train.exe

Write-Host "  -> forest_baseline_predict.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() $coreObjs "obj/main_predict_baseline.o" -o forest_baseline_predict.exe

Write-Host "  -> forest_optimized_predict.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() $coreObjs "obj/main_predict_optimized.o" -o forest_optimized_predict.exe

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Pronto! Executaveis gerados." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green