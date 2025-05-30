# PowerShell script for training all models with GPU enforcement
param(
    [string]$DataCSV = "data/metadata.csv",
    [string]$ImgRoot = "data/danbooru_images/",
    [string]$OutputDir = "experiments"
)

Write-Host "========================================"
Write-Host "   DANBOORU GPU TRAINING PIPELINE"
Write-Host "========================================"
Write-Host "GPU: RTX 3070 Laptop (8GB VRAM)"
Write-Host "RAM: 32GB"
Write-Host "CPU: Ryzen 9"
Write-Host "========================================"

# GPU Check
Write-Host "Checking GPU availability..."
try {
    $gpuCheckCommand = "import torch; print('CUDA_STATUS:' + str(torch.cuda.is_available())); print('GPU_NAME:' + str(torch.cuda.get_device_name(0)) if torch.cuda.is_available() else 'None'); print('GPU_MEMORY:' + str(torch.cuda.get_device_properties(0).total_memory // 1024**3) + 'GB' if torch.cuda.is_available() else '0GB')"
    $gpuCheckOutput = python -c $gpuCheckCommand
    Write-Host $gpuCheckOutput -ForegroundColor Cyan
    
    # Проверяем есть ли CUDA_STATUS:True в выводе
    $cudaAvailable = $gpuCheckOutput -match "CUDA_STATUS:True"
    
    if (-not $cudaAvailable) {
        Write-Host "❌ CUDA not available! Please install PyTorch with CUDA support." -ForegroundColor Red
        Write-Host "Exiting..." -ForegroundColor Red
        exit 1
    } else {
        Write-Host "✅ GPU Ready for training!" -ForegroundColor Green
    }
} catch {
    Write-Warning "Could not check GPU status: $($_.Exception.Message)"
    Write-Host "Exiting..." -ForegroundColor Red
    exit 1
}

Write-Host "========================================"

# Check prerequisites
if (-not (Test-Path $DataCSV)) {
    Write-Error "❌ CSV file not found: $DataCSV"
    exit 1
}

if (-not (Test-Path $ImgRoot)) {
    Write-Error "❌ Images directory not found: $ImgRoot"
    exit 1
}

if (-not (Test-Path "src/main_train.py")) {
    Write-Error "❌ Training script not found: src/main_train.py"
    exit 1
}

# Create timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logFile = "gpu_training_pipeline_$timestamp.log"

# Common parameters optimized for GPU
$commonParams = @(
    "--data_csv", $DataCSV,
    "--img_root", $ImgRoot,
    "--output_dir", $OutputDir,
    "--use_amp",
    "--num_workers", "6",
    "--top_k_tags", "1000",
    "--epochs", "25"
)

# Counters
$successCount = 0
$failedCount = 0

Write-Host "Pipeline log: $logFile" -ForegroundColor Yellow
Write-Host "Timestamp: $timestamp" -ForegroundColor Yellow
Write-Host ""

# Log function
function Write-Log {
    param([string]$Message)
    $timeStr = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timeStr] $Message"
    Write-Host $logMessage
    Add-Content -Path $logFile -Value $logMessage
}

function Show-EstimatedTime {
    param([int]$ModelNumber, [int]$TotalModels)
    
    $estimatedTimes = @{
        1 = "1.0-1.5 hours"   # ResNet34
        2 = "1.5-2.0 hours"   # ResNet50  
        3 = "1.5-2.0 hours"   # EfficientNet-B0
        4 = "2.0-2.5 hours"   # EfficientNet-B2
        5 = "2.5-3.0 hours"   # EfficientNet-B3
        6 = "3.0-4.0 hours"   # EfficientNet-B4
        7 = "2.0-2.5 hours"   # ConvNeXt-Tiny
        8 = "3.0-4.0 hours"   # Swin Transformer
        9 = "3.0-4.0 hours"   # Vision Transformer
        10 = "1.5-2.0 hours"  # MobileViT
    }
    
    Write-Host "⏱️ Estimated training time: $($estimatedTimes[$ModelNumber])" -ForegroundColor Magenta
}

Write-Log "Starting GPU training pipeline with 10 models"

# All 10 models configuration optimized for RTX 3070 Laptop (8GB VRAM)
$models = @(
    @{
        Name = "ResNet34"
        ModelName = "resnet34" 
        BatchSize = 64
        LR = "1e-3"
        ImageSize = 224
        Scheduler = "cosine"
        Description = "Fast baseline - warmup model"
        Priority = "High"
    },
    @{
        Name = "ResNet50"
        ModelName = "resnet50"
        BatchSize = 48
        LR = "8e-4" 
        ImageSize = 256
        Scheduler = "cosine"
        WeightDecay = "0.01"
        Description = "Standard baseline model"
        Priority = "High"
    },
    @{
        Name = "EfficientNet-B0"
        ModelName = "efficientnet_b0"
        BatchSize = 64
        LR = "1e-3"
        ImageSize = 224
        Scheduler = "cosine"
        Optimizer = "adamw"
        Description = "Efficient small model"
        Priority = "High"
    },
    @{
        Name = "EfficientNet-B2"
        ModelName = "efficientnet_b2"
        BatchSize = 32
        LR = "8e-4"
        ImageSize = 260
        Scheduler = "cosine"
        AccumSteps = 2
        Description = "Balanced efficiency model"
        Priority = "High"
    },
    @{
        Name = "EfficientNet-B3"
        ModelName = "efficientnet_b3"
        BatchSize = 24
        LR = "6e-4"
        ImageSize = 300
        Scheduler = "cosine"
        AccumSteps = 3
        WeightDecay = "0.02"
        Description = "High quality model"
        Priority = "Medium"
    },
    @{
        Name = "EfficientNet-B4"
        ModelName = "efficientnet_b4"
        BatchSize = 16
        LR = "5e-4"
        ImageSize = 320
        Scheduler = "cosine"
        AccumSteps = 4
        WeightDecay = "0.03"
        Epochs = 30
        Description = "Maximum quality model"
        Priority = "Medium"
    },
    @{
        Name = "ConvNeXt-Tiny"
        ModelName = "convnext_tiny"
        BatchSize = 28
        LR = "4e-4"
        ImageSize = 224
        Scheduler = "cosine"
        WeightDecay = "0.05"
        AccumSteps = 2
        Description = "Modern CNN architecture"
        Priority = "Medium"
    },
    @{
        Name = "Swin Transformer"
        ModelName = "swin_t"
        BatchSize = 16
        LR = "3e-4"
        ImageSize = 224
        Scheduler = "cosine"
        WeightDecay = "0.05"
        AccumSteps = 4
        Epochs = 30
        Description = "Vision Transformer for images"
        Priority = "Low"
    },
    @{
        Name = "Vision Transformer"
        ModelName = "vit_s_16"
        BatchSize = 20
        LR = "3e-4"
        ImageSize = 224
        Scheduler = "cosine"
        WeightDecay = "0.1"
        AccumSteps = 3
        Epochs = 35
        Description = "Classic ViT architecture"
        Priority = "Low"
    },
    @{
        Name = "MobileViT"
        ModelName = "mobilevit_s"
        BatchSize = 32
        LR = "1e-3"
        ImageSize = 256
        Scheduler = "step"
        AccumSteps = 2
        Description = "Mobile-optimized architecture"
        Priority = "Medium"
    }
)

$totalModels = $models.Length
Write-Host "🚀 Will train $totalModels models with GPU acceleration" -ForegroundColor Green
Write-Host "📊 Estimated total time: 20-25 hours" -ForegroundColor Yellow
Write-Host "💡 Tip: You can safely leave this running overnight!" -ForegroundColor Cyan
Write-Host ""

# Ask for confirmation
$confirmation = Read-Host "Ready to start training all $totalModels models? This will take 20+ hours. Type 'YES' to continue"
if ($confirmation -ne 'YES') {
    Write-Host "Training cancelled by user." -ForegroundColor Yellow
    exit 0
}

Write-Host ""
Write-Host "🎯 Starting marathon training session..." -ForegroundColor Green
Write-Host ""

# Train models
for ($i = 0; $i -lt $models.Length; $i++) {
    $model = $models[$i]
    $modelNum = $i + 1
    
    Write-Host ""
    Write-Log "========================================"
    Write-Log "MODEL $modelNum/$($models.Length): $($model.Name) ($($model.Description))"
    Write-Log "========================================"
    
    # Show priority and estimated time
    Write-Host "🎯 Priority: $($model.Priority)" -ForegroundColor $(if($model.Priority -eq "High"){"Green"}elseif($model.Priority -eq "Medium"){"Yellow"}else{"Red"})
    Show-EstimatedTime -ModelNumber $modelNum -TotalModels $totalModels
    
    # Pre-training GPU status
    Write-Host "🔍 Pre-training GPU status:"
    try {
        $gpuStatusCmd = "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB, Available: {torch.cuda.is_available()}') if torch.cuda.is_available() else print('CUDA not available')"
        $gpuStatus = python -c $gpuStatusCmd
        Write-Host $gpuStatus -ForegroundColor Cyan
    } catch {
        Write-Host "Could not get GPU status" -ForegroundColor Yellow
    }
    
    # Prepare arguments
    $args = $commonParams.Clone()
    
    # Override epochs if specified in model
    if ($model.ContainsKey("Epochs")) {
        # Remove default epochs and add model-specific epochs
        $newArgs = @()
        $skipNext = $false
        for ($j = 0; $j -lt $args.Length; $j++) {
            if ($skipNext) {
                $skipNext = $false
                continue
            }
            if ($args[$j] -eq "--epochs") {
                $skipNext = $true
                continue
            }
            $newArgs += $args[$j]
        }
        $args = $newArgs + @("--epochs", $model.Epochs)
    }
    
    $args += @(
        "--model_name", $model.ModelName,
        "--batch_size", $model.BatchSize,
        "--lr", $model.LR,
        "--image_size", $model.ImageSize,
        "--scheduler_name", $model.Scheduler,
        "--experiment_name", "$($model.ModelName)_full_$timestamp"
    )
    
    # Additional parameters
    if ($model.ContainsKey("WeightDecay")) { $args += @("--weight_decay", $model.WeightDecay) }
    if ($model.ContainsKey("AccumSteps")) { $args += @("--accumulation_steps", $model.AccumSteps) }
    if ($model.ContainsKey("Optimizer")) { $args += @("--optimizer_name", $model.Optimizer) }
    
    # Debug: show command that will be executed
    Write-Host "Command: python src/main_train.py $($args[2..($args.Length-1)] -join ' ')" -ForegroundColor Gray
    
    # Start training with performance monitoring
    Write-Log "🚀 Starting GPU training for $($model.Name)"
    Write-Host "⚡ Expected GPU utilization: 70-90%" -ForegroundColor Green
    Write-Host "💾 Expected VRAM usage: 3-7GB" -ForegroundColor Green
    Write-Host "🕐 Start time: $(Get-Date -Format 'HH:mm:ss')" -ForegroundColor Magenta
    
    # Execute with detailed error handling and timing
    try {
        $startTime = Get-Date
        Write-Host "🏃‍♂️ Model $modelNum/$totalModels training started..." -ForegroundColor Green
        
        $process = Start-Process -FilePath "python" -ArgumentList (@("src/main_train.py") + $args) -Wait -PassThru -NoNewWindow
        
        $endTime = Get-Date
        $duration = $endTime - $startTime
        $durationStr = $duration.ToString('hh\:mm\:ss')
        
        if ($process.ExitCode -eq 0) {
            Write-Log "✅ $($model.Name) - SUCCESS (Duration: $durationStr)"
            Write-Host "✅ $($model.Name) - SUCCESS" -ForegroundColor Green
            Write-Host "⏱️ Training time: $durationStr" -ForegroundColor Green
            $successCount++
            
            # Post-training cleanup and GPU status
            try {
                $cleanupCmd = "import torch; torch.cuda.empty_cache(); print(f'GPU Memory after cleanup: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB, Peak: {torch.cuda.max_memory_allocated(0)/1024**3:.1f}GB') if torch.cuda.is_available() else print('No CUDA')"
                $cleanupStatus = python -c $cleanupCmd
                Write-Host "🧹 $cleanupStatus" -ForegroundColor Blue
            } catch {
                Write-Host "Could not cleanup GPU memory" -ForegroundColor Yellow
            }
            
            # Calculate remaining time estimate
            if ($i -lt ($models.Length - 1)) {
                $avgTimePerModel = $duration.TotalMinutes / 1
                $remainingModels = $models.Length - $modelNum
                $estimatedRemainingHours = ($avgTimePerModel * $remainingModels) / 60
                Write-Host "📈 Estimated remaining time: $([math]::Round($estimatedRemainingHours, 1)) hours" -ForegroundColor Magenta
            }
            
        } else {
            Write-Log "❌ $($model.Name) - FAILED (exit code: $($process.ExitCode), Duration: $durationStr)"
            Write-Host "❌ $($model.Name) - FAILED (exit code: $($process.ExitCode))" -ForegroundColor Red
            Write-Host "⏱️ Failed after: $durationStr" -ForegroundColor Red
            $failedCount++
        }
    } catch {
        Write-Log "❌ $($model.Name) - EXCEPTION: $($_.Exception.Message)"
        Write-Host "❌ $($model.Name) - EXCEPTION: $($_.Exception.Message)" -ForegroundColor Red
        $failedCount++
    }
    
    # Progress summary
    Write-Host ""
    Write-Host "📊 Progress: $modelNum/$totalModels completed | ✅ Success: $successCount | ❌ Failed: $failedCount" -ForegroundColor Cyan
    
    # Brief pause between models (except for the last one)
    if ($i -lt ($models.Length - 1)) {
        Write-Host "⏳ Cooling down for 20 seconds before next model..." -ForegroundColor Yellow
        Start-Sleep -Seconds 20
        Write-Host ""
    }
}

# Final comprehensive report
$pipelineEndTime = Get-Date
Write-Host ""
Write-Host "🏁 TRAINING PIPELINE COMPLETED! 🏁" -ForegroundColor Green
Write-Host ""

Write-Log "========================================"
Write-Log "FINAL COMPREHENSIVE TRAINING REPORT"
Write-Log "========================================"
Write-Log "Total models: $($models.Length)"
Write-Log "Successfully trained: $successCount"
Write-Log "Failed: $failedCount"
Write-Log "Success rate: $([math]::Round(($successCount / $models.Length) * 100, 1))%"
Write-Log "Pipeline completed at: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')"
Write-Log "========================================"

# Success analysis
if ($successCount -eq $models.Length) {
    Write-Host "🎉 PERFECT! ALL $($models.Length) MODELS TRAINED SUCCESSFULLY!" -ForegroundColor Green
    Write-Host "🥇 You now have a complete model zoo for comparison!" -ForegroundColor Green
} elseif ($successCount -ge ($models.Length * 0.8)) {
    Write-Host "🎊 EXCELLENT! $successCount/$($models.Length) models trained successfully!" -ForegroundColor Green
    Write-Host "📈 Success rate: $([math]::Round(($successCount / $models.Length) * 100, 1))%" -ForegroundColor Green
} elseif ($successCount -ge ($models.Length * 0.5)) {
    Write-Host "👍 GOOD! $successCount/$($models.Length) models trained successfully!" -ForegroundColor Yellow
    Write-Host "💪 You have a solid foundation for comparison!" -ForegroundColor Yellow
} elseif ($successCount -gt 0) {
    Write-Host "⚠️ PARTIAL SUCCESS: $successCount/$($models.Length) models completed" -ForegroundColor Yellow
    Write-Host "🔄 Consider re-running failed models individually" -ForegroundColor Yellow
} else {
    Write-Host "❌ ALL MODELS FAILED - Please check your setup" -ForegroundColor Red
    Write-Host "🔧 Check logs for debugging information" -ForegroundColor Red
}

Write-Host ""
Write-Host "📂 Results Summary:" -ForegroundColor Cyan
Write-Host "   📁 Experiments folder: $OutputDir" -ForegroundColor White
Write-Host "   📜 Pipeline log: $logFile" -ForegroundColor White
Write-Host "   🏷️ Timestamp: $timestamp" -ForegroundColor White

# List successful experiments
if ($successCount -gt 0) {
    Write-Host ""
    Write-Host "✅ Successfully trained models:" -ForegroundColor Green
    for ($i = 0; $i -lt $models.Length; $i++) {
        $model = $models[$i]
        $expDir = "$OutputDir/$($model.ModelName)_full_$timestamp"
        if (Test-Path $expDir) {
            $bestModelPath = "$expDir/best_model.pth"
            if (Test-Path $bestModelPath) {
                Write-Host "   🎯 $($model.Name): $expDir" -ForegroundColor Green
            }
        }
    }
}

# List failed models for retry
if ($failedCount -gt 0) {
    Write-Host ""
    Write-Host "❌ Failed models (for manual retry):" -ForegroundColor Red
    for ($i = 0; $i -lt $models.Length; $i++) {
        $model = $models[$i]
        $expDir = "$OutputDir/$($model.ModelName)_full_$timestamp"
        if (-not (Test-Path "$expDir/best_model.pth")) {
            Write-Host "   🔄 $($model.Name) - Retry command:" -ForegroundColor Yellow
            Write-Host "      python src/main_train.py --model_name $($model.ModelName) --data_csv $DataCSV --img_root $ImgRoot --batch_size $($model.BatchSize) --lr $($model.LR) --use_amp" -ForegroundColor Gray
        }
    }
}

Write-Host ""
Write-Host "🔍 Final system status:"
try {
    $finalGpuCmd = "import torch; print(f'GPU Memory: {torch.cuda.memory_allocated(0)/1024**3:.1f}GB, Peak Session: {torch.cuda.max_memory_allocated(0)/1024**3:.1f}GB, Available: {torch.cuda.is_available()}') if torch.cuda.is_available() else print('CUDA not available')"
    $finalGpuStatus = python -c $finalGpuCmd
    Write-Host $finalGpuStatus -ForegroundColor Cyan
} catch {
    Write-Host "Could not get final GPU status" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "🎉 Training marathon completed! Great job!" -ForegroundColor Magenta
Write-Host "📊 Check your results and compare model performances!" -ForegroundColor Cyan
Write-Host ""

# Final pause
Write-Host "Press any key to exit..." -ForegroundColor White
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")