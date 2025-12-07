# ============================================================
#   Script de Treinamento Completo
#   Treina modelos Baseline e Otimizado em 3 tamanhos
# ============================================================

Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "  Treinamento Completo: Baseline vs Otimizado" -ForegroundColor Cyan
Write-Host "  Tamanhos: 10k, 100k, 500k amostras" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# Criar pastas necessarias
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}

if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

$dataset = "covertype_dataset.csv"
$sizes = @(10000, 100000, 500000)

# Arquivo de log geral
$logFile = "results/training_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
"Treinamento iniciado em: $(Get-Date)" | Out-File $logFile

Write-Host "Log sendo salvo em: $logFile" -ForegroundColor Green
Write-Host ""

# ============================================================
# Loop pelos tamanhos de dataset
# ============================================================

foreach ($size in $sizes) {
    $sizeLabel = "{0}k" -f ($size / 1000)
    
    Write-Host "========================================================" -ForegroundColor Yellow
    Write-Host "  Treinando com $sizeLabel amostras" -ForegroundColor Yellow
    Write-Host "========================================================" -ForegroundColor Yellow
    Write-Host ""
    
    # ------------------------------------------------------------
    # 1. BASELINE
    # ------------------------------------------------------------
    
    $baselineModel = "models/baseline_$($sizeLabel).model"
    $baselineLog = "results/baseline_$($sizeLabel)_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    
    Write-Host "[BASELINE $sizeLabel] Iniciando treinamento..." -ForegroundColor Cyan
    $startTime = Get-Date
    
    ./forest_baseline_train.exe $dataset $size 1 $baselineModel | Tee-Object -FilePath $baselineLog
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host "[BASELINE $sizeLabel] Concluido em $([math]::Round($duration, 2)) segundos" -ForegroundColor Green
    Write-Host ""
    
    # Log geral
    "BASELINE ${sizeLabel} : $([math]::Round($duration, 2))s" | Out-File $logFile -Append
    
    # ------------------------------------------------------------
    # 2. OTIMIZADO
    # ------------------------------------------------------------
    
    $optimizedModel = "models/optimized_$($sizeLabel).model"
    $optimizedLog = "results/optimized_$($sizeLabel)_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
    
    Write-Host "[OTIMIZADO $sizeLabel] Iniciando treinamento..." -ForegroundColor Cyan
    $startTime = Get-Date
    
    ./forest_optimized_train.exe $dataset $size 1 $optimizedModel | Tee-Object -FilePath $optimizedLog
    
    $endTime = Get-Date
    $duration = ($endTime - $startTime).TotalSeconds
    
    Write-Host "[OTIMIZADO $sizeLabel] Concluido em $([math]::Round($duration, 2)) segundos" -ForegroundColor Green
    Write-Host ""
    
    # Log geral
    "OTIMIZADO ${sizeLabel}: $([math]::Round($duration, 2))s" | Out-File $logFile -Append
    "" | Out-File $logFile -Append
}

# ============================================================
# Resumo final
# ============================================================

Write-Host "========================================================" -ForegroundColor Green
Write-Host "  TREINAMENTO COMPLETO!" -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Modelos salvos em: models/" -ForegroundColor White
Write-Host "Logs salvos em: results/" -ForegroundColor White
Write-Host ""

Get-ChildItem -Path "models/*.model" | ForEach-Object {
    $sizeMB = [math]::Round($_.Length / 1MB, 2)
    Write-Host "  -> $($_.Name) ($sizeMB MB)" -ForegroundColor Cyan
}

Write-Host ""
Write-Host "Log completo: $logFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para fazer predicoes, use:" -ForegroundColor White
Write-Host "  ./forest_baseline_predict.exe models/baseline_10k.model $dataset 1000" -ForegroundColor Gray
Write-Host "  ./forest_optimized_predict.exe models/optimized_10k.model $dataset 1000" -ForegroundColor Gray
Write-Host ""
