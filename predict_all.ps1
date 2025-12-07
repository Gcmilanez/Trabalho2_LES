# ============================================================
#   Script de Predição Completa - Baseline vs Otimizado
#   Testa predição em modelos já treinados
# ============================================================

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Testes de Predicao: Baseline vs Otimizado" -ForegroundColor Cyan
Write-Host "  Carrega modelos treinados e mede tempo de predicao" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

# Criar pasta de resultados se não existir
if (-not (Test-Path "results")) {
    New-Item -ItemType Directory -Path "results" | Out-Null
}

# Configuração dos datasets e modelos
$tests = @(
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

# Arquivo de log
$logFile = "results/prediction_log_$(Get-Date -Format 'yyyyMMdd_HHmmss').txt"
"Testes de Predicao iniciados em: $(Get-Date)" | Out-File $logFile
"" | Out-File $logFile -Append

Write-Host "Log sendo salvo em: $logFile" -ForegroundColor Green
Write-Host ""

# Número de execuções para média
$num_runs = 5

# ============================================================
# Loop pelos datasets
# ============================================================

foreach ($test in $tests) {
    $datasetName = $test.Name
    $datasetFile = $test.File
    
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host "  DATASET: $($datasetName.ToUpper())" -ForegroundColor Yellow
    Write-Host "  Arquivo: $datasetFile" -ForegroundColor Yellow
    Write-Host "============================================================" -ForegroundColor Yellow
    Write-Host ""
    
    "DATASET: $($datasetName.ToUpper())" | Out-File $logFile -Append
    "Arquivo: $datasetFile" | Out-File $logFile -Append
    "" | Out-File $logFile -Append
    
    # Loop pelos tamanhos
    foreach ($size in $test.Sizes) {
        # Formatar label do tamanho (igual ao train_all.ps1)
        $sizeLabel = ""
        if ($size -lt 1000) {
            # Menor que 1000: usar formato 0,5k
            $sizeLabel = "$([string]::Format('{0:N1}', $size/1000).Replace('.', ','))k"
        } elseif ($size % 1000 -eq 0) {
            # Múltiplo exato de 1000: usar 1k, 10k, etc (sem vírgula)
            $sizeLabel = "$($size/1000)k"
        } else {
            # Não é múltiplo de 1000: usar formato com vírgula (45,222k)
            $sizeLabel = "$([string]::Format('{0:N0}', $size).Replace('.', ','))k"
        }
        
        Write-Host "========================================================" -ForegroundColor Cyan
        Write-Host "  [$datasetName] Testando predicao com $size amostras" -ForegroundColor Cyan
        Write-Host "========================================================" -ForegroundColor Cyan
        Write-Host ""
        
        # Caminhos dos modelos (usar mesmo formato que train_all.ps1)
        $baselineModel = "models/baseline_${datasetName}_${sizeLabel}.model"
        $optimizedModel = "models/optimized_${datasetName}_${sizeLabel}.model"
        
        # Verificar se modelos existem
        if (-not (Test-Path $baselineModel)) {
            Write-Host "⚠ Modelo baseline não encontrado: $baselineModel" -ForegroundColor Yellow
            continue
        }
        
        if (-not (Test-Path $optimizedModel)) {
            Write-Host "⚠ Modelo otimizado não encontrado: $optimizedModel" -ForegroundColor Yellow
            continue
        }
        
        # ========================================
        # BASELINE PREDICT
        # ========================================
        Write-Host "[BASELINE $datasetName $sizeLabel] Testando predicao..." -ForegroundColor White
        
        $startTime = Get-Date
        $output = & ".\forest_baseline_predict.exe" $datasetFile $baselineModel $size $num_runs 2>&1 | Out-String
        $endTime = Get-Date
        $elapsed = ($endTime - $startTime).TotalSeconds
        
        # Extrair tempo médio de predição do output
        $predTime = "N/A"
        $accuracy = "N/A"
        if ($output -match "Tempo Predicao Medio \(ms\)\s+([\d.]+)") {
            $predTime = [math]::Round([double]$Matches[1], 2)
        }
        if ($output -match "Acuracia Media \(%\)\s+([\d.]+)") {
            $accuracy = [math]::Round([double]$Matches[1], 2)
        }
        
        Write-Host "[BASELINE $datasetName $sizeLabel] Tempo predicao: ${predTime}ms, Acuracia: ${accuracy}%" -ForegroundColor Green
        "BASELINE $datasetName ${sizeLabel}: ${predTime}ms (${accuracy}% acc)" | Out-File $logFile -Append
        
        # ========================================
        # OPTIMIZED PREDICT
        # ========================================
        Write-Host "[OTIMIZADO $datasetName $sizeLabel] Testando predicao..." -ForegroundColor White
        
        $startTime = Get-Date
        $output = & ".\forest_optimized_predict.exe" $datasetFile $optimizedModel $size $num_runs 2>&1 | Out-String
        $endTime = Get-Date
        $elapsed = ($endTime - $startTime).TotalSeconds
        
        # Extrair tempo médio de predição do output
        $predTime = "N/A"
        $accuracy = "N/A"
        if ($output -match "Tempo Predicao Medio \(ms\)\s+([\d.]+)") {
            $predTime = [math]::Round([double]$Matches[1], 2)
        }
        if ($output -match "Acuracia Media \(%\)\s+([\d.]+)") {
            $accuracy = [math]::Round([double]$Matches[1], 2)
        }
        
        Write-Host "[OTIMIZADO $datasetName $sizeLabel] Tempo predicao: ${predTime}ms, Acuracia: ${accuracy}%" -ForegroundColor Green
        "OTIMIZADO $datasetName ${sizeLabel}: ${predTime}ms (${accuracy}% acc)" | Out-File $logFile -Append
        "" | Out-File $logFile -Append
        
        Write-Host ""
    }
    
    Write-Host ""
}

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  Testes de predicao concluidos!" -ForegroundColor Green
Write-Host "  Log salvo em: $logFile" -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
