@echo off
chcp 65001 > nul
echo ========================================
echo    БЫСТРОЕ ТЕСТИРОВАНИЕ 3 МОДЕЛЕЙ
echo ========================================
echo Для RTX 3070 - сокращенное обучение
echo ========================================

set DATA_CSV=data/metadata.csv
set IMG_ROOT=data/danbooru_images/
set COMMON_PARAMS=--data_csv %DATA_CSV% --img_root %IMG_ROOT% --use_amp --num_workers 4 --top_k_tags 500 --epochs 5

for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /value') do if not "%%I"=="" set datetime=%%I
set TIMESTAMP=%datetime:~0,8%_%datetime:~8,6%

echo [%time%] Тестирование ResNet34...
python src/main_train.py --model_name resnet34 %COMMON_PARAMS% --batch_size 64 --experiment_name test_resnet34_%TIMESTAMP%

echo [%time%] Тестирование EfficientNet-B0...
python src/main_train.py --model_name efficientnet_b0 %COMMON_PARAMS% --batch_size 48 --experiment_name test_effb0_%TIMESTAMP%

echo [%time%] Тестирование ConvNeXt...
python src/main_train.py --model_name convnext_tiny %COMMON_PARAMS% --batch_size 32 --experiment_name test_convnext_%TIMESTAMP%

echo Быстрое тестирование завершено!
pause