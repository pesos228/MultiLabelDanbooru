Write-Host "=== QUICK MODEL TEST ===" -ForegroundColor Green
Write-Host "Testing single ResNet34 model"

# Check prerequisites
if (-not (Test-Path "data/metadata.csv")) {
    Write-Error "data/metadata.csv not found!"
    exit 1
}

if (-not (Test-Path "data/danbooru_images")) {
    Write-Error "data/danbooru_images folder not found!"  
    exit 1
}

if (-not (Test-Path "src/main_train.py")) {
    Write-Error "src/main_train.py not found!"
    exit 1
}

# Create timestamp
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
Write-Host "Timestamp: $timestamp" -ForegroundColor Yellow

# Test training
Write-Host "Starting training test..." -ForegroundColor Cyan

$args = @(
    "src/main_train.py",
    "--model_name", "resnet34",
    "--data_csv", "data/metadata.csv", 
    "--img_root", "data/danbooru_images/",
    "--batch_size", "16",
    "--epochs", "2",
    "--top_k_tags", "50",
    "--use_amp",
    "--experiment_name", "test_single_$timestamp"
)

$process = Start-Process -FilePath "python" -ArgumentList $args -Wait -PassThru -NoNewWindow

if ($process.ExitCode -eq 0) {
    Write-Host "SUCCESS: Test completed!" -ForegroundColor Green
} else {
    Write-Host "ERROR: Test failed with code $($process.ExitCode)" -ForegroundColor Red
    Write-Host "Common issues:" -ForegroundColor Yellow
    Write-Host "- Missing data files"
    Write-Host "- CUDA/PyTorch installation issues" 
    Write-Host "- Insufficient GPU memory"
}

Read-Host "Press Enter to continue"