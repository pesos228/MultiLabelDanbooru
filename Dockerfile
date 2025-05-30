# Используйте официальный PyTorch образ
FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

# Рабочая директория
WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements и установка Python зависимостей
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копирование исходного кода
COPY src/ ./src/
COPY *.py ./

# Создание директорий для монтирования
RUN mkdir -p /app/data /app/experiments

# Установка переменных окружения
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Порт (если нужен)
EXPOSE 8000

# Команда по умолчанию
CMD ["python", "-c", "print('Danbooru Tagger Container Ready!')"]