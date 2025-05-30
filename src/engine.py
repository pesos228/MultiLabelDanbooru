"""
Training and evaluation functions with GPU optimization and monitoring
"""
import logging
from typing import List, Optional, Tuple, Dict
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
try:
    # Для новых версий PyTorch
    from torch.amp import GradScaler, autocast
except ImportError:
    # Для старых версий PyTorch
    from torch.cuda.amp import GradScaler, autocast

from tqdm import tqdm
import numpy as np

import torchmetrics
from sklearn.metrics import f1_score, hamming_loss

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[GradScaler] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    accumulation_steps: int = 1,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Train model for one epoch with GPU optimization
    """
    model.train()

    # CHECK MODEL IS ON GPU
    model_device = next(model.parameters()).device
    if model_device != device:
        logger.warning(f"Model is on {model_device}, moving to {device}")
        model = model.to(device)

    total_loss = 0.0
    num_batches = len(dataloader)
    num_samples = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")

    optimizer.zero_grad()

    # GPU memory monitoring
    if device.type == 'cuda':
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated(device) / 1024**3
        logger.info(f"Initial GPU memory: {initial_memory:.2f}GB")

    for batch_idx, (images, targets) in enumerate(pbar):
        # FORCE GPU LOADING
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # CHECK DATA IS ON GPU
        if device.type == 'cuda' and images.device != device:
            logger.error(f"Images not on GPU! Device: {images.device}")
            break

        batch_size = images.size(0)
        num_samples += batch_size

        # Forward pass with mixed precision
        if scaler is not None and device.type == 'cuda':
            # Use autocast context manager
            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = criterion(outputs, targets)
                loss = loss / accumulation_steps
        else:
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss = loss / accumulation_steps

        # Backward pass
        if scaler is not None and device.type == 'cuda':
            scaler.scale(loss).backward()
        else:
            loss.backward()

        total_loss += loss.item() * accumulation_steps

        # Optimizer step (every accumulation_steps or last batch)
        if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == num_batches:
            if scaler is not None and device.type == 'cuda':
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            optimizer.zero_grad()

        # GPU monitoring every 100 batches
        if device.type == 'cuda' and batch_idx % 100 == 0:
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(device) / 1024**3

            if batch_idx == 0:
                logger.info(f"GPU memory usage: {current_memory:.2f}GB (max: {max_memory:.2f}GB)")

        # Update progress bar
        avg_loss = total_loss / (batch_idx + 1)
        postfix = {
            'loss': f'{avg_loss:.4f}',
            'lr': f'{optimizer.param_groups[0]["lr"]:.2e}'
        }

        if device.type == 'cuda':
            postfix['gpu'] = f"{torch.cuda.memory_allocated(device) / 1024**3:.1f}GB"

        pbar.set_postfix(postfix)

    # Scheduler step (if exists)
    if scheduler is not None:
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            pass  # will call after validation
        else:
            scheduler.step()

    avg_loss = total_loss / num_batches

    # Final GPU memory info
    if device.type == 'cuda':
        final_memory = torch.cuda.memory_allocated(device) / 1024**3
        max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
        logger.info(f"Epoch {epoch} GPU memory - Final: {final_memory:.2f}GB, Peak: {max_memory:.2f}GB")

    return {
        'loss': avg_loss,
        'lr': optimizer.param_groups[0]['lr'],
        'num_samples': num_samples
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    num_classes: int,
    tag_vocab: List[str],
    threshold: float = 0.5,
    epoch: int = 0
) -> Dict[str, float]:
    """
    Evaluate model on validation data with GPU optimization
    """
    model.eval()

    # CHECK MODEL IS ON GPU
    if next(model.parameters()).device != device:
        model = model.to(device)

    total_loss = 0.0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    num_samples = 0

    # Progress bar
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")

    # Clear GPU cache
    if device.type == 'cuda':
        torch.cuda.empty_cache()

    for images, targets in pbar:
        # FORCE GPU LOADING
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        batch_size = images.size(0)
        num_samples += batch_size

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, targets)
        total_loss += loss.item()

        # Convert to probabilities
        probabilities = torch.sigmoid(outputs)

        # Save for metrics computation
        all_probabilities.append(probabilities.cpu())
        all_targets.append(targets.cpu())

        # Binarize predictions
        predictions = (probabilities > threshold).float()
        all_predictions.append(predictions.cpu())

        # Update progress bar
        avg_loss = total_loss / len(pbar)
        postfix = {'val_loss': f'{avg_loss:.4f}'}

        if device.type == 'cuda':
            postfix['gpu'] = f"{torch.cuda.memory_allocated(device) / 1024**3:.1f}GB"

        pbar.set_postfix(postfix)

    # Combine all batches
    all_predictions = torch.cat(all_predictions, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_probabilities = torch.cat(all_probabilities, dim=0).numpy()

    # Compute metrics
    avg_loss = total_loss / len(dataloader)

    # F1 scores
    micro_f1 = f1_score(all_targets, all_predictions, average='micro', zero_division=0)
    macro_f1 = f1_score(all_targets, all_predictions, average='macro', zero_division=0)

    # Hamming Loss
    hamming = hamming_loss(all_targets, all_predictions)

    # Additional metrics
    # Subset accuracy (exact match)
    subset_accuracy = np.mean(np.all(all_targets == all_predictions, axis=1))

    # Mean Average Precision
    try:
        from sklearn.metrics import average_precision_score
        mean_ap = average_precision_score(all_targets, all_probabilities, average='macro')
    except:
        mean_ap = 0.0

    # Number of active predictions
    avg_predictions_per_sample = np.mean(np.sum(all_predictions, axis=1))
    avg_targets_per_sample = np.mean(np.sum(all_targets, axis=1))

    metrics = {
        'loss': avg_loss,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'hamming_loss': hamming,
        'subset_accuracy': subset_accuracy,
        'mean_ap': mean_ap,
        'avg_predictions_per_sample': avg_predictions_per_sample,
        'avg_targets_per_sample': avg_targets_per_sample,
        'num_samples': num_samples
    }

    return metrics


def log_metrics(metrics: Dict[str, float], prefix: str = "", epoch: int = None):
    """Log metrics"""
    log_str = f"{prefix}"
    if epoch is not None:
        log_str += f" Epoch {epoch}"
    
    log_str += " | "
    
    for key, value in metrics.items():
        if key == 'num_samples':
            continue
        if isinstance(value, float):
            log_str += f"{key}: {value:.4f} | "
        else:
            log_str += f"{key}: {value} | "
    
    logger.info(log_str.rstrip(" | "))