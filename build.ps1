# ============================================================
#   Script de Compilação - Random Forest Baseline vs Otimizada
# ============================================================

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Compilando Random Forest (Baseline + Otimizada)" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Criar diretório obj se não existir
if (-not (Test-Path "obj")) {
    New-Item -ItemType Directory -Path "obj" | Out-Null
}

$CXX = "g++"
$CXXFLAGS = "-std=c++17 -O3 -Wall -Wextra -march=native"

# ============================================================
# Compilar objetos base
# ============================================================

Write-Host "[1/7] Compilando DecisionTree.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c DecisionTree.cpp -o obj/DecisionTree.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[2/7] Compilando RandomForestBaseline.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c RandomForestBaseline.cpp -o obj/RandomForestBaseline.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[3/7] Compilando RandomForestOptimized.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c RandomForestOptimized.cpp -o obj/RandomForestOptimized.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[4/7] Compilando main_forest_baseline.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_forest_baseline.cpp -o obj/main_forest_baseline.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[5/7] Compilando main_forest_optimized.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_forest_optimized.cpp -o obj/main_forest_optimized.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[6/7] Compilando main_predict_baseline.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_predict_baseline.cpp -o obj/main_predict_baseline.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "[7/7] Compilando main_predict_optimized.cpp..." -ForegroundColor Yellow
& $CXX $CXXFLAGS.Split() -c main_predict_optimized.cpp -o obj/main_predict_optimized.o
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""

# ============================================================
# Linkar executáveis
# ============================================================

Write-Host "Linkando executaveis..." -ForegroundColor Green
Write-Host ""

Write-Host "  -> forest_baseline_train.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() obj/DecisionTree.o obj/RandomForestBaseline.o obj/main_forest_baseline.o -o forest_baseline_train.exe
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "  -> forest_optimized_train.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() obj/DecisionTree.o obj/RandomForestOptimized.o obj/main_forest_optimized.o -o forest_optimized_train.exe
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "  -> forest_baseline_predict.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() obj/DecisionTree.o obj/RandomForestBaseline.o obj/main_predict_baseline.o -o forest_baseline_predict.exe
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host "  -> forest_optimized_predict.exe" -ForegroundColor White
& $CXX $CXXFLAGS.Split() obj/DecisionTree.o obj/RandomForestOptimized.o obj/main_predict_optimized.o -o forest_optimized_predict.exe
if ($LASTEXITCODE -ne 0) { exit 1 }

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  Compilacao concluida com sucesso!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Executaveis criados:" -ForegroundColor Cyan
Write-Host "  -> forest_baseline_train.exe" -ForegroundColor White
Write-Host "  -> forest_optimized_train.exe" -ForegroundColor White
Write-Host "  -> forest_baseline_predict.exe" -ForegroundColor White
Write-Host "  -> forest_optimized_predict.exe" -ForegroundColor White
Write-Host ""
