Write-Host "=== QUICK SINGLE MODEL TEST ===" -ForegroundColor Green

# GPU Check
$gpuCheck = python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
Write-Host $gpuCheck

# Test single ResNet34
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$args = @(
    "src/main_train.py",
    "--model_name", "resnet34",
    "--data_csv", "data/metadata.csv",
    "--img_root", "data/danbooru_images/",
    "--batch_size", "32",
    "--epochs", "3",
    "--top_k_tags", "100",  # Smaller for testing
    "--use_amp",
    "--experiment_name", "quick_test_$timestamp"
)

Write-Host "Starting quick test..." -ForegroundColor Cyan
Write-Host "Command: python $($args -join ' ')" -ForegroundColor Gray

$process = Start-Process -FilePath "python" -ArgumentList $args -Wait -PassThru -NoNewWindow

if ($process.ExitCode -eq 0) {
    Write-Host "✅ Quick test SUCCESS!" -ForegroundColor Green
} else {
    Write-Host "❌ Quick test FAILED (exit code: $($process.ExitCode))" -ForegroundColor Red
}

Read-Host "Press Enter to continue"