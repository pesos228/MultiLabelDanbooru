"""
Utilities for danbooru tagger project
"""
import logging
import random
import numpy as np
import torch
from pathlib import Path
from typing import Optional


def set_seed(seed: int = 42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_logging(log_file: Optional[Path] = None, level: int = logging.INFO):
    """Setup logging system with UTF-8 encoding"""
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Configure root logger
    logger = logging.getLogger()
    logger.setLevel(level)
    logger.handlers.clear()

    # Console handler with UTF-8 encoding
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Force UTF-8 encoding for Windows
    if hasattr(console_handler.stream, 'reconfigure'):
        try:
            console_handler.stream.reconfigure(encoding='utf-8')
        except:
            pass

    logger.addHandler(console_handler)

    # File handler with UTF-8 encoding
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def count_parameters(model: torch.nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config_dict: dict, save_path: Path):
    """Save experiment configuration"""
    import json

    # Convert Path objects to strings for JSON serialization
    config_copy = {}
    for key, value in config_dict.items():
        if isinstance(value, Path):
            config_copy[key] = str(value)
        else:
            config_copy[key] = value

    with open(save_path, 'w', encoding='utf-8') as f:
        json.dump(config_copy, f, indent=2, ensure_ascii=False)


def check_gpu():
    """Check GPU availability and properties"""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        current_gpu = torch.cuda.current_device()
        gpu_name = torch.cuda.get_device_name(current_gpu)
        gpu_memory = torch.cuda.get_device_properties(current_gpu).total_memory // 1024**3

        print(f"GPU Available: YES")
        print(f"GPU Count: {gpu_count}")
        print(f"Current GPU: {current_gpu}")
        print(f"GPU Name: {gpu_name}")
        print(f"GPU Memory: {gpu_memory} GB")
        return True
    else:
        print("GPU Available: NO")
        return False


def force_gpu_usage():
    """Force GPU usage and error if not available"""
    if not torch.cuda.is_available():
        print("ERROR: CUDA is not available!")
        print("Please install PyTorch with CUDA support:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        exit(1)

    device = torch.device('cuda:0')
    torch.cuda.set_device(device)
    return device