# ============================================================
#   Script de Treinamento Completo - Multiplos Datasets
#   Treina modelos Baseline e Otimizado com Adult e MNIST
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Treinamento Completo: Baseline vs Otimizado" -ForegroundColor Cyan
Write-Host "  Datasets: OptDigits (1.8k) + Adult (45k) + Skin (245k)" -ForegroundColor Cyan
Write-Host "  Tamanhos variados por dataset" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Criar pastas necessarias
if (-not (Test-Path "models")) {
    New-Item -ItemType Directory -Path "models" | Out-Null
}

if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# Configuracao dos datasets
$datasets = @(
    @{
        Name = "optdigits"
        File = "optdigits.csv"
        Sizes = @(500, 1000, 1797)
    },
    @{
        Name = "adult"
        File = "adult_dataset.csv"
        Sizes = @(1000, 10000, 45222)
    },
    @{
        Name = "skin"
        File = "skin_segmentation.csv"
        Sizes = @(1000, 10000, 245057)
    }
)

# Arquivo de log geral
$logFile = "results/training_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
"Treinamento iniciado em: $(Get-Date)" | Out-File $logFile
"" | Out-File $logFile -Append

Write-Host "Log sendo salvo em: $logFile" -ForegroundColor Green
Write-Host ""

# ============================================================
# Loop pelos datasets
# ============================================================

foreach ($ds in $datasets) {
    $datasetName = $ds.Name
    $datasetFile = $ds.File
    $sizes = $ds.Sizes
    
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host "  DATASET: $($datasetName.ToUpper())" -ForegroundColor Magenta
    Write-Host "  Arquivo: $datasetFile" -ForegroundColor Magenta
    Write-Host "============================================================" -ForegroundColor Magenta
    Write-Host ""
    
    "DATASET: $($datasetName.ToUpper())" | Out-File $logFile -Append
    "Arquivo: $datasetFile" | Out-File $logFile -Append
    "" | Out-File $logFile -Append
    
    # ============================================================
    # Loop pelos tamanhos de dataset
    # ============================================================
    
    foreach ($size in $sizes) {
        $sizeLabel = "{0}k" -f ($size / 1000)
        
        Write-Host "========================================================" -ForegroundColor Yellow
        Write-Host "  [$($datasetName.ToUpper())] Treinando com $sizeLabel amostras" -ForegroundColor Yellow
        Write-Host "========================================================" -ForegroundColor Yellow
        Write-Host ""
        
        # ------------------------------------------------------------
        # 1. BASELINE
        # ------------------------------------------------------------
        
        $baselineModel = "models/baseline_$($datasetName)_$($sizeLabel).model"
        $baselineLog = "results/baseline_$($datasetName)_$($sizeLabel)_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
        
        Write-Host "[BASELINE $datasetName $sizeLabel] Iniciando treinamento..." -ForegroundColor Cyan
        $startTime = Get-Date
        
        ./forest_baseline_train.exe $datasetFile $size 1 $baselineModel | Tee-Object -FilePath $baselineLog
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        Write-Host "[BASELINE $datasetName $sizeLabel] Concluido em $([math]::Round($duration, 2)) segundos" -ForegroundColor Green
        Write-Host ""
        
        # Log geral
        "BASELINE $datasetName ${sizeLabel}: $([math]::Round($duration, 2))s" | Out-File $logFile -Append
        
        # ------------------------------------------------------------
        # 2. OTIMIZADO
        # ------------------------------------------------------------
        
        $optimizedModel = "models/optimized_$($datasetName)_$($sizeLabel).model"
        $optimizedLog = "results/optimized_$($datasetName)_$($sizeLabel)_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"
        
        Write-Host "[OTIMIZADO $datasetName $sizeLabel] Iniciando treinamento..." -ForegroundColor Cyan
        $startTime = Get-Date
        
        ./forest_optimized_train.exe $datasetFile $size 1 $optimizedModel | Tee-Object -FilePath $optimizedLog
        
        $endTime = Get-Date
        $duration = ($endTime - $startTime).TotalSeconds
        
        Write-Host "[OTIMIZADO $datasetName $sizeLabel] Concluido em $([math]::Round($duration, 2)) segundos" -ForegroundColor Green
        Write-Host ""
        
        # Log geral
        "OTIMIZADO $datasetName ${sizeLabel}: $([math]::Round($duration, 2))s" | Out-File $logFile -Append
        "" | Out-File $logFile -Append
    }
    
    # Separador entre datasets
    "============================================================" | Out-File $logFile -Append
    "" | Out-File $logFile -Append
}

# ============================================================
# Resumo final
# ============================================================

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  TREINAMENTO COMPLETO!" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
Write-Host ""
Write-Host "Modelos salvos em: models/" -ForegroundColor White
Write-Host "Logs salvos em: results/" -ForegroundColor White
Write-Host ""

# Listar modelos por dataset
Write-Host "MODELOS TREINADOS:" -ForegroundColor Cyan
Write-Host ""

foreach ($ds in $datasets) {
    Write-Host "  $($ds.Name.ToUpper()):" -ForegroundColor Yellow
    Get-ChildItem -Path "models/*$($ds.Name)*.model" | ForEach-Object {
        $sizeMB = [math]::Round($_.Length / 1MB, 2)
        Write-Host "    -> $($_.Name) ($sizeMB MB)" -ForegroundColor White
    }
    Write-Host ""
}

Write-Host "Log completo: $logFile" -ForegroundColor Yellow
Write-Host ""
Write-Host "Para fazer predicoes, use:" -ForegroundColor White
Write-Host "  ./forest_baseline_predict.exe models/baseline_adult_10k.model adult_dataset.csv 1000" -ForegroundColor Gray
Write-Host "  ./forest_optimized_predict.exe models/optimized_mnist_10k.model mnist_dataset.csv 1000" -ForegroundColor Gray
Write-Host ""
