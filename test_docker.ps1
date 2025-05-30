# test_docker.ps1
$ImageName = "yourusername/danbooru-tagger:latest"

Write-Host "🐳 Testing Danbooru Tagger Docker Image" -ForegroundColor Green
Write-Host "========================================"

# Build
Write-Host "Building image..." -ForegroundColor Yellow
docker build -t $ImageName .

# GPU Test
Write-Host "Testing GPU..." -ForegroundColor Yellow
docker run --rm --gpus all $ImageName python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Quick training test
Write-Host "Quick training test..." -ForegroundColor Yellow
docker run --rm --gpus all `
    -v ${PWD}/data:/app/data `
    -v ${PWD}/experiments:/app/experiments `
    $ImageName `
    python src/main_train.py `
    --model_name resnet34 `
    --data_csv data/metadata.csv `
    --img_root data/danbooru_images/ `
    --batch_size 16 `
    --epochs 2 `
    --top_k_tags 50 `
    --use_amp `
    --experiment_name docker_powershell_test

Write-Host "✅ Test completed!" -ForegroundColor Green