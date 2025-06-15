# Danbooru Tagger Project
Система для экспериментов по обучению различных моделей классификации изображений на датасете Danbooru для мультилейбл классификации тегов.

## Особенности
- 🏗️ **Модульная архитектура** - легко расширяемый и поддерживаемый код
- 🚀 **Множество архитектур** - ResNet, EfficientNet, ConvNeXt, Swin Transformer, Vision Transformer, MobileViT
- 🎯 **Мультилейбл классификация** - предсказание множества тегов для каждого изображения
- ⚡ **Современные техники** - Mixed Precision, Gradient Accumulation, различные оптимизаторы и schedulers
- 📊 **Подробные метрики** - F1-score (micro/macro), Hamming Loss, Mean AP и другие
- 🔧 **Гибкая конфигурация** - настройка через аргументы командной строки
- 📝 **Автоматическое логирование** - сохранение всех параметров и результатов экспериментов

## Поддерживаемые модели
- ResNet34, ResNet50
- EfficientNet-B0, B2, B3, B4
- ConvNeXt-Tiny
- Swin Transformer (Tiny)
- Vision Transformer (Small)
- MobileViT (Small)

Все модели обучаются **с нуля** (без предобученных весов).

## Быстрый старт

### 1. Настройка окружения
```bash
# Создание virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate     # Windows

# Установка зависимостей
pip install -r requirements.txt
```
Альтернативно, с conda:
```bash
conda create -n tagger_env python=3.9 -y
conda activate tagger_env
pip install -r requirements.txt
```

### 2. Подготовка данных
Структура данных:
```
data/
├── danbooru_images/
│   ├── 5997041.jpg
│   ├── 6001234.jpg
│   └── ...
└── metadata.csv
```
Формат `metadata.csv`:
```csv
id,filename,archive,tags
20512,5997041.jpg,data-0004.zip,"1girl, apron, bangs, blue eyes, blue hair, ..."
```

### 3. Запуск обучения
Базовый пример:
```bash
python src/main_train.py \
    --model_name resnet50 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --epochs 20 \
    --batch_size 64 \
    --top_k_tags 1000 \
    --experiment_name resnet50_baseline
```
Пример с расширенными настройками:
```bash
python src/main_train.py \
    --model_name efficientnet_b2 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --epochs 30 \
    --batch_size 32 \
    --image_size 288 \
    --lr 5e-4 \
    --optimizer_name adamw \
    --scheduler_name cosine \
    --weight_decay 0.01 \
    --top_k_tags 2000 \
    --min_tag_frequency 20 \
    --use_amp \
    --accumulation_steps 2 \
    --experiment_name efficientnet_b2_advanced
```

### 4. Предсказание тегов для новых изображений

После обучения модели вы можете использовать её для предсказания тегов на новых изображениях:

```bash
# Базовый пример предсказания
python src/predict_sample.py \
    --checkpoint_path "experiments/resnet50_baseline/best_model.pth" \
    --image_paths "path/to/your/image.jpg" \
    --threshold 0.5 \
    --top_n_probs 15
```

**Пример с реальными путями:**
```bash
# Предсказание для одного изображения
python src/predict_sample.py \
    --checkpoint_path "C:\Users\PC\Documents\university\AI\MultiLabelDanbooru\experiments\ResNet50_4090_test_run_on_volume_20250530_200726\best_model.pth" \
    --image_paths "C:\Users\PC\Downloads\s-l1200.jpg" \
    --threshold 0.5 \
    --top_n_probs 15

# Предсказание для нескольких изображений
python src/predict_sample.py \
    --checkpoint_path "experiments/efficientnet_b3_amp/best_model.pth" \
    --image_paths "image1.jpg" "image2.jpg" "image3.jpg" \
    --threshold 0.3 \
    --top_n_probs 20

# Более консервативный порог для точных предсказаний
python src/predict_sample.py \
    --checkpoint_path "experiments/swin_tiny_exp1/best_model.pth" \
    --image_paths "anime_image.jpg" \
    --threshold 0.7 \
    --top_n_probs 10
```

**Параметры предсказания:**

| Параметр           | Описание                                              | Пример значения |
|--------------------|-------------------------------------------------------|-----------------|
| `--checkpoint_path`| Путь к сохранённой модели (.pth файл)                | "experiments/model/best_model.pth" |
| `--image_paths`    | Путь(и) к изображению(ям) для предсказания           | "image.jpg" или несколько файлов |
| `--threshold`      | Порог вероятности для отбора тегов (0.0-1.0)         | 0.5 |
| `--top_n_probs`    | Количество топовых тегов для вывода                   | 15 |

**Пример вывода:**
```
Предсказание для изображения: s-l1200.jpg
Найдено тегов выше порога 0.5: 8

Топ-15 предсказанных тегов:
1girl         0.892
long_hair     0.834
blue_eyes     0.781
school_uniform 0.723
smile         0.678
blonde_hair   0.645
sitting       0.591
indoors       0.532
...
```

### 5. Параметры запуска
Основные параметры:

| Параметр              | Описание                   | По умолчанию |
|-----------------------|----------------------------|--------------|
| `--model_name`        | Архитектура модели         | Обязательный |
| `--data_csv`          | Путь к CSV с метаданными   | Обязательный |
| `--img_root`          | Корневая папка изображений | Обязательный |
| `--epochs`            | Количество эпох            | 20           |
| `--batch_size`        | Размер батча               | 32           |
| `--lr`                | Learning rate              | 1e-3         |
| `--top_k_tags`        | Топ тегов в словаре        | 1000         |
| `--image_size`        | Размер изображения         | 256          |
| `--use_amp`           | Mixed precision            | False        |
| `--experiment_name`   | Имя эксперимента           | auto         |

Полный список параметров: `python src/main_train.py --help`

## Готовые команды для запуска разных моделей
Скопируйте и запустите следующие команды для обучения различных архитектур:

### ResNet модели
```bash
# ResNet34 - быстрая модель для экспериментов
python src/main_train.py \
    --model_name resnet34 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --batch_size 64 \
    --experiment_name resnet34_exp1

# ResNet50 - стандартная baseline модель
python src/main_train.py \
    --model_name resnet50 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --batch_size 48 \
    --lr 5e-4 \
    --weight_decay 0.01 \
    --experiment_name resnet50_baseline
```

### EfficientNet модели
```bash
# EfficientNet-B0 - компактная и эффективная
python src/main_train.py \
    --model_name efficientnet_b0 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 224 \
    --batch_size 64 \
    --lr 1e-3 \
    --use_amp \
    --experiment_name efficientnet_b0_amp

# EfficientNet-B2 - баланс скорости и качества
python src/main_train.py \
    --model_name efficientnet_b2 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 260 \
    --batch_size 40 \
    --lr 8e-4 \
    --use_amp \
    --experiment_name efficientnet_b2_260px

# EfficientNet-B3 - высокое качество с AMP
python src/main_train.py \
    --model_name efficientnet_b3 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 320 \
    --batch_size 32 \
    --lr 6e-4 \
    --use_amp \
    --accumulation_steps 2 \
    --experiment_name efficientnet_b3_amp

# EfficientNet-B4 - максимальное качество
python src/main_train.py \
    --model_name efficientnet_b4 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 380 \
    --batch_size 24 \
    --lr 5e-4 \
    --use_amp \
    --accumulation_steps 3 \
    --scheduler_name cosine \
    --experiment_name efficientnet_b4_380px
```

### Transformer модели
```bash
# Swin Transformer - отлично для изображений
python src/main_train.py \
    --model_name swin_t \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 224 \
    --batch_size 16 \
    --lr 5e-4 \
    --weight_decay 0.05 \
    --scheduler_name cosine \
    --use_amp \
    --experiment_name swin_tiny_exp1

# Vision Transformer - классический ViT
python src/main_train.py \
    --model_name vit_s_16 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 224 \
    --batch_size 24 \
    --lr 3e-4 \
    --weight_decay 0.1 \
    --scheduler_name cosine \
    --use_amp \
    --experiment_name vit_small_exp1
```

### Современные архитектуры
```bash
# ConvNeXt - современная CNN архитектура
python src/main_train.py \
    --model_name convnext_tiny \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 224 \
    --batch_size 32 \
    --lr 4e-4 \
    --weight_decay 0.05 \
    --scheduler_name cosine \
    --use_amp \
    --experiment_name convnext_tiny_exp1

# MobileViT - эффективная мобильная архитектура
python src/main_train.py \
    --model_name mobilevit_s \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --image_size 256 \
    --batch_size 48 \
    --lr 1e-3 \
    --scheduler_name step \
    --use_amp \
    --experiment_name mobilevit_s_exp1
```

### Команды для разных размеров датасета
```bash
# Для небольших экспериментов (топ-500 тегов)
python src/main_train.py \
    --model_name resnet34 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --top_k_tags 500 \
    --min_tag_frequency 5 \
    --epochs 15 \
    --batch_size 64 \
    --experiment_name quick_test_500tags

# Для полных экспериментов (топ-2000 тегов)
python src/main_train.py \
    --model_name efficientnet_b3 \
    --data_csv data/metadata.csv \
    --img_root data/danbooru_images/ \
    --top_k_tags 2000 \
    --min_tag_frequency 20 \
    --epochs 50 \
    --batch_size 32 \
    --image_size 320 \
    --use_amp \
    --experiment_name full_experiment_2k_tags
```

## Рекомендации по запуску

### Для начинающих экспериментов:
- Используйте **ResNet34** или **EfficientNet-B0** для быстрого тестирования
- Начните с `--top_k_tags 500` и `--epochs 10`
- Увеличивайте `batch_size` до максимально возможного для вашей GPU

### Для серьезных экспериментов:
- **EfficientNet-B3/B4** или **Swin Transformer** для лучшего качества
- Включайте `--use_amp` для экономии памяти
- Используйте `--accumulation_steps` если нужен большой effective batch size
- Настройте `scheduler_name cosine` для лучшей сходимости

### Настройки для разных GPU:

| GPU Memory | Рекомендуемые настройки                                     |
|------------|-------------------------------------------------------------|
| 6-8 GB     | `batch_size=16-24`, `image_size=224`, `--use_amp`           |
| 10-12 GB   | `batch_size=32-48`, `image_size=256-288`, `--use_amp`         |
| 16+ GB     | `batch_size=64+`, `image_size=320+`, опционально `--use_amp` |

### Рекомендации по предсказанию:
- **Threshold 0.3-0.4** - для получения большего количества тегов
- **Threshold 0.5-0.6** - сбалансированный вариант
- **Threshold 0.7+** - только самые уверенные предсказания
- **top_n_probs 10-20** - оптимальное количество для анализа

## Структура проекта
```
danbooru_tagger_project/
├── src/                    # Исходный код
│   ├── dataset.py         # Класс датасета и трансформации
│   ├── models.py          # Определения архитектур моделей
│   ├── engine.py          # Функции обучения и оценки
│   ├── main_train.py      # Основной скрипт обучения
│   ├── predict_sample.py  # Скрипт для предсказания тегов
│   └── utils.py          # Вспомогательные утилиты
├── data/                  # Данные
├── experiments/           # Результаты экспериментов
├── notebooks/            # Jupyter notebooks для анализа
├── requirements.txt      # Зависимости Python
└── README.md            # Документация
```

## Результаты экспериментов
После обучения в папке `experiments/{experiment_name}/` создаются:
- `best_model.pth` - лучшая модель по Macro F1
- `last_model.pth` - модель с последней эпохи
- `config.json` - конфигурация эксперимента
- `history.json` - история обучения
- `train.log` - подробные логи

## Метрики качества
Для оценки моделей используются:
- **Micro F1-score** - общая производительность
- **Macro F1-score** - сбалансированная оценка по всем классам
- **Hamming Loss** - процент неправильных меток
- **Subset Accuracy** - точное совпадение всех меток
- **Mean Average Precision** - средняя точность по классам

