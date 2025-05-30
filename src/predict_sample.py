import argparse
import json
from pathlib import Path
import random

import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np

# Предполагается, что models.py и dataset.py находятся в том же каталоге или доступны через sys.path
from models import get_model # Используем ваш обновленный models.py
from dataset import get_val_transforms # Трансформации для валидации/инференса

def load_checkpoint(checkpoint_path: Path, device: torch.device):
    """Загружает чекпоинт модели и связанные данные."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    args_dict = checkpoint['args']
    # Преобразуем обратно в Namespace для удобства, если нужно
    # args = argparse.Namespace(**args_dict) 

    model_name = args_dict['model_name']
    # num_classes уже будет в модели из checkpoint['tag_vocab']
    
    tag_vocab = checkpoint['tag_vocab']
    num_classes = len(tag_vocab)

    model = get_model(model_name, num_classes=num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"Модель '{model_name}' загружена с чекпоинта: {checkpoint_path}")
    print(f"Количество классов (тегов): {num_classes}")
    return model, tag_vocab, args_dict

def predict_image(
    model: torch.nn.Module,
    image_path: Path,
    transform: transforms.Compose,
    device: torch.device,
    tag_vocab: list,
    threshold: float = 0.5, # Порог для бинаризации предсказаний
    top_n_probs: int = 10    # Сколько тегов с наивысшими вероятностями показать
):
    """Делает предсказание для одного изображения."""
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Ошибка загрузки изображения {image_path}: {e}")
        return None, None

    img_tensor = transform(img).unsqueeze(0).to(device) # Добавляем batch_size=1

    with torch.no_grad():
        logits = model(img_tensor)
        probabilities = torch.sigmoid(logits).squeeze(0) # Убираем batch_size=1

    # Теги, предсказанные с вероятностью выше порога
    predicted_indices_threshold = (probabilities > threshold).nonzero(as_tuple=True)[0]
    predicted_tags_threshold = [tag_vocab[i] for i in predicted_indices_threshold.cpu().tolist()]
    predicted_probs_threshold = probabilities[predicted_indices_threshold].cpu().tolist()

    # Топ-N тегов по вероятности (независимо от порога)
    top_probs, top_indices = torch.topk(probabilities, top_n_probs)
    top_predicted_tags = [tag_vocab[i] for i in top_indices.cpu().tolist()]
    top_predicted_probs = top_probs.cpu().tolist()

    print(f"\n--- Предсказания для: {image_path.name} ---")
    if predicted_tags_threshold:
        print(f"\nТеги с вероятностью > {threshold}:")
        for tag, prob in zip(predicted_tags_threshold, predicted_probs_threshold):
            print(f"  - {tag}: {prob:.4f}")
    else:
        print(f"\nНет тегов с вероятностью > {threshold}.")

    print(f"\nТоп-{top_n_probs} тегов по вероятности:")
    for tag, prob in zip(top_predicted_tags, top_predicted_probs):
        print(f"  - {tag}: {prob:.4f}")
    
    return predicted_tags_threshold, top_predicted_tags


def main():
    parser = argparse.ArgumentParser(description="Инференс модели Danbooru Tagger на одном или нескольких изображениях.")
    parser.add_argument('--checkpoint_path', type=str, required=True,
                        help="Путь к файлу чекпоинта (.pth)")
    parser.add_argument('--image_paths', type=str, nargs='+', required=True,
                        help="Пути к изображениям для предсказания (один или несколько)")
    parser.add_argument('--threshold', type=float, default=0.3, # Снизим порог для наглядности
                        help="Порог вероятности для отнесения тега к предсказанным")
    parser.add_argument('--top_n_probs', type=int, default=10,
                        help="Количество тегов с наивысшими вероятностями для вывода")
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help="Устройство для инференса (cuda/cpu)")

    script_args = parser.parse_args()

    device = torch.device(script_args.device)

    model, tag_vocab, train_args = load_checkpoint(Path(script_args.checkpoint_path), device)
    
    # Получаем image_size из сохраненных аргументов обучения
    image_size = train_args.get('image_size', 256) # 256 как дефолт, если вдруг нет в старых конфигах
    print(f"Используется image_size: {image_size} из конфига обучения.")
    
    # Трансформации для инференса (обычно как для валидации)
    # Убедитесь, что get_val_transforms определен в вашем dataset.py или импортируйте его правильно
    try:
        from dataset import get_val_transforms
        val_transform = get_val_transforms(image_size=image_size)
    except ImportError:
        print("ПРЕДУПРЕЖДЕНИЕ: Не удалось импортировать get_val_transforms из dataset.py. Используются стандартные трансформации.")
        val_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    for img_path_str in script_args.image_paths:
        img_path = Path(img_path_str)
        if not img_path.exists():
            print(f"Файл изображения не найден: {img_path}")
            continue
        predict_image(
            model,
            img_path,
            val_transform,
            device,
            tag_vocab,
            threshold=script_args.threshold,
            top_n_probs=script_args.top_n_probs
        )

if __name__ == '__main__':
    main()