# test_docker.ps1
$ImageName = "yourusername/danbooru-tagger:latest"
Write-Host "🐳 Testing Danbooru Tagger Docker Image for RTX 3070" -ForegroundColor Green
Write-Host "========================================"
# # Build (можно закомментировать, если образ уже собран)
# Write-Host "Building image..." -ForegroundColor Yellow
# docker build -t $ImageName .

# GPU Test
Write-Host "Testing GPU..." -ForegroundColor Yellow
docker run --rm --gpus all $ImageName python -c "import torch; print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

# Swin-T training test for RTX 3070
Write-Host "Swin-T training test for RTX 3070..." -ForegroundColor Yellow
docker run --rm --gpus all `
    -v ${PWD}/data:/app/data `
    -v ${PWD}/experiments:/app/experiments `
    $ImageName `
    python /app/src/main_train.py `
    --data_csv /app/data/metadata.csv `
    --img_root /app/data/danbooru_images/ `
    --output_dir /app/experiments/ `
    --model_name swin_t `
    --batch_size 16 `
    --lr 5e-4 `
    --image_size 224 `
    --scheduler_name cosine `
    --weight_decay 0.05 `
    --use_amp `
    --num_workers 4 `
    --top_k_tags 1000 `
    --epochs 30 `
    --accumulation_steps 2 `
    --experiment_name SwinT_3070_bs16acc2_is224_30ep_docker_test

Write-Host "✅ Test completed!" -ForegroundColor Green