"""
Main training script for multi-label classification models with GPU optimization
"""
import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import create_train_val_datasets
from models import get_model, get_available_models
from engine import train_one_epoch, evaluate, log_metrics
from utils import set_seed, setup_logging, count_parameters, save_config, force_gpu_usage, check_gpu

logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train models for multi-label Danbooru classification')

    # Data
    parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to CSV file with metadata')
    parser.add_argument('--img_root', type=str, required=True,
                        help='Path to root directory with images')

    # Model
    parser.add_argument('--model_name', type=str, required=True,
                        choices=get_available_models(),
                        help='Model architecture name')

    # Dataset parameters
    parser.add_argument('--top_k_tags', type=int, default=1000,
                        help='Number of most frequent tags for vocabulary')
    parser.add_argument('--min_tags_per_image', type=int, default=1,
                        help='Minimum number of relevant tags per image')
    parser.add_argument('--min_tag_frequency', type=int, default=10,
                        help='Minimum tag frequency for vocabulary inclusion')

    # Training parameters
    parser.add_argument('--image_size', type=int, default=256,
                        help='Image size')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-2,
                        help='Weight decay')

    # Optimizer and scheduler
    parser.add_argument('--optimizer_name', type=str, default='adamw',
                        choices=['adamw', 'sgd'],
                        help='Optimizer name')
    parser.add_argument('--scheduler_name', type=str, default='cosine',
                        choices=['cosine', 'step', 'plateau', 'none'],
                        help='Scheduler name')

    # Training techniques
    parser.add_argument('--use_amp', action='store_true',
                        help='Use Automatic Mixed Precision')
    parser.add_argument('--accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')

    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of DataLoader workers')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device (cuda/cpu/auto)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    # Output
    parser.add_argument('--output_dir', type=str, default='./experiments',
                        help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                        help='Experiment name')

    # Validation and saving
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation data fraction')
    parser.add_argument('--save_best_only', action='store_true',
                        help='Save only best model')

    return parser.parse_args()


def create_optimizer(model: nn.Module, args) -> torch.optim.Optimizer:
    """Create optimizer"""
    if args.optimizer_name == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.lr,
            weight_decay=args.weight_decay
        )
    elif args.optimizer_name == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            weight_decay=args.weight_decay
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer_name}")

    return optimizer


def create_scheduler(optimizer: torch.optim.Optimizer, args, steps_per_epoch: int):
    """Create learning rate scheduler"""
    if args.scheduler_name == 'none':
        return None
    elif args.scheduler_name == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs * steps_per_epoch,
            eta_min=args.lr * 0.01
        )
    elif args.scheduler_name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.epochs // 3,
            gamma=0.1
        )
    elif args.scheduler_name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=0.5,
            patience=3,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {args.scheduler_name}")

    return scheduler


def create_amp_scaler(use_amp: bool, device: torch.device):
    """Create AMP scaler with version compatibility"""
    if not use_amp or device.type != 'cuda':
        return None

    try:
        # Try new PyTorch API first
        from torch.amp import GradScaler
        scaler = GradScaler('cuda')
        logger.info("Using new torch.amp.GradScaler API")
        return scaler
    except ImportError:
        try:
            # Fall back to old API
            from torch.cuda.amp import GradScaler
            scaler = GradScaler()
            logger.info("Using legacy torch.cuda.amp.GradScaler API")
            return scaler
        except Exception as e:
            logger.warning(f"Could not create GradScaler: {e}. Using FP32 training.")
            return None


def main():
    args = get_args()

    # Set seed
    set_seed(args.seed)

    # FORCE GPU USAGE
    print("="*80)
    print("GPU SETUP AND VERIFICATION")
    print("="*80)

    if not check_gpu():
        print("ERROR: CUDA not available!")
        print("Please install PyTorch with CUDA support:")
        print("pip uninstall torch torchvision torchaudio")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        exit(1)

    device = force_gpu_usage()
    print(f"FORCED GPU USAGE: {device}")

    # Test GPU with simple computation
    try:
        test_tensor = torch.randn(1000, 1000).to(device)
        result = torch.matmul(test_tensor, test_tensor)
        gpu_memory = torch.cuda.memory_allocated(device) / 1024**3
        print(f"GPU test passed. Memory usage: {gpu_memory:.2f}GB")
        del test_tensor, result
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"GPU test failed: {e}")
        exit(1)

    print("="*80)

    # Create experiment name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.experiment_name is None:
        experiment_name = f"{args.model_name}_{timestamp}"
    else:
        experiment_name = f"{args.experiment_name}_{timestamp}"

    # Create experiment directory (output_dir already includes base folder name)
    experiment_dir = Path(args.output_dir)
    # Add timestamp to the output_dir path itself
    if not str(args.output_dir).endswith(timestamp):
        experiment_dir = Path(f"{args.output_dir}_{timestamp}")

    # Create experiment directory
    experiment_dir = Path(args.output_dir) / experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = experiment_dir / 'train.log'
    setup_logging(log_file)

    logger.info("="*80)
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Device: {device}")
    logger.info(f"Experiment directory: {experiment_dir}")
    logger.info("="*80)

    # Save configuration
    config_dict = vars(args).copy()
    config_dict['device'] = str(device)
    config_dict['experiment_dir'] = str(experiment_dir)
    config_dict['pytorch_version'] = torch.__version__
    save_config(config_dict, experiment_dir / 'config.json')

    # Create datasets
    logger.info("Creating datasets...")
    train_dataset, val_dataset = create_train_val_datasets(
        csv_path=Path(args.data_csv),
        root_img_dir=Path(args.img_root),
        image_size=args.image_size,
        top_k_tags=args.top_k_tags,
        min_tag_frequency=args.min_tag_frequency,
        min_tags_per_image=args.min_tags_per_image,
        val_size=args.val_size,
        random_state=args.seed
    )

    num_classes = train_dataset.num_classes
    logger.info(f"Number of classes: {num_classes}")
    logger.info(f"Train dataset size: {len(train_dataset)}")
    logger.info(f"Val dataset size: {len(val_dataset)}")

    # Create DataLoaders with optimized settings for GPU
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,  # IMPORTANT FOR GPU
        drop_last=True,
        persistent_workers=True if args.num_workers > 0 else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,  # IMPORTANT FOR GPU
        persistent_workers=True if args.num_workers > 0 else False
    )

    logger.info("DataLoaders created with pin_memory=True for GPU optimization")

    # Create model
    logger.info("Creating model...")
    model = get_model(args.model_name, num_classes)

    # FORCE MODEL TO GPU
    model = model.to(device)
    logger.info(f"Model moved to device: {next(model.parameters()).device}")
    logger.info(f"Number of parameters: {count_parameters(model):,}")

    # Verify model is on GPU
    if next(model.parameters()).device != device:
        logger.error("MODEL NOT ON GPU!")
        exit(1)

    # Loss function
    criterion = nn.BCEWithLogitsLoss()

    # Optimizer
    optimizer = create_optimizer(model, args)

    # Scheduler
    scheduler = create_scheduler(optimizer, args, len(train_loader))

    # AMP Scaler with version compatibility
    scaler = create_amp_scaler(args.use_amp, device)
    if scaler is not None:
        logger.info("Using Automatic Mixed Precision (AMP)")
    else:
        logger.info("Using FP32 training (no AMP)")

    # Variables for tracking best model
    best_macro_f1 = 0.0
    best_epoch = 0

    # Metrics history
    train_history = []
    val_history = []

    logger.info("Starting training...")
    logger.info("="*80)

    # Training loop
    for epoch in range(1, args.epochs + 1):
        epoch_start_time = datetime.now()

        # Clear GPU cache before each epoch
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Training
        try:
            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                scheduler=scheduler if args.scheduler_name != 'plateau' else None,
                accumulation_steps=args.accumulation_steps,
                epoch=epoch
            )
        except Exception as e:
            logger.error(f"Training failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Validation
        try:
            val_metrics = evaluate(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                device=device,
                num_classes=num_classes,
                tag_vocab=train_dataset.tag_vocab,
                epoch=epoch
            )
        except Exception as e:
            logger.error(f"Validation failed at epoch {epoch}: {e}")
            import traceback
            traceback.print_exc()
            break

        # Scheduler step for ReduceLROnPlateau
        if scheduler is not None and args.scheduler_name == 'plateau':
            scheduler.step(val_metrics['macro_f1'])

        # Logging
        epoch_time = datetime.now() - epoch_start_time

        logger.info(f"\nEpoch {epoch}/{args.epochs} ({epoch_time})")
        log_metrics(train_metrics, "TRAIN")
        log_metrics(val_metrics, "VAL  ")

        # GPU memory info
        if device.type == 'cuda':
            current_memory = torch.cuda.memory_allocated(device) / 1024**3
            max_memory = torch.cuda.max_memory_allocated(device) / 1024**3
            logger.info(f"GPU Memory: {current_memory:.2f}GB (peak: {max_memory:.2f}GB)")

        # Save metrics history
        train_history.append(train_metrics)
        val_history.append(val_metrics)

        # Save best model
        current_macro_f1 = val_metrics['macro_f1']
        if current_macro_f1 > best_macro_f1:
            best_macro_f1 = current_macro_f1
            best_epoch = epoch

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'best_macro_f1': best_macro_f1,
                'val_metrics': val_metrics,
                'args': vars(args),
                'tag_vocab': train_dataset.tag_vocab,
                'pytorch_version': torch.__version__
            }

            torch.save(checkpoint, experiment_dir / 'best_model.pth')
            logger.info(f"Saved best model (Macro F1: {best_macro_f1:.4f})")

        # Save last model (if not best only)
        if not args.save_best_only:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'scaler_state_dict': scaler.state_dict() if scaler else None,
                'val_metrics': val_metrics,
                'args': vars(args),
                'tag_vocab': train_dataset.tag_vocab,
                'pytorch_version': torch.__version__
            }
            torch.save(checkpoint, experiment_dir / 'last_model.pth')

        # Save intermediate history every 5 epochs
        if epoch % 5 == 0:
            history = {
                'train': train_history,
                'val': val_history,
                'best_epoch': best_epoch,
                'best_macro_f1': best_macro_f1,
                'current_epoch': epoch
            }

            with open(experiment_dir / 'history_temp.json', 'w') as f:
                # Convert numpy types to regular for JSON
                def convert_numpy(obj):
                    if hasattr(obj, 'item'):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj

                history_json = convert_numpy(history)
                json.dump(history_json, f, indent=2)

    # Save final training history
    history = {
        'train': train_history,
        'val': val_history,
        'best_epoch': best_epoch,
        'best_macro_f1': best_macro_f1,
        'total_epochs': len(train_history),
        'experiment_completed': True
    }

    with open(experiment_dir / 'history.json', 'w') as f:
        # Convert numpy types to regular for JSON
        def convert_numpy(obj):
            if hasattr(obj, 'item'):
                return obj.item()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj

        history_json = convert_numpy(history)
        json.dump(history_json, f, indent=2)

    logger.info("="*80)
    logger.info("Training completed!")
    logger.info(f"Best result: Epoch {best_epoch}, Macro F1: {best_macro_f1:.4f}")
    logger.info(f"Models saved in: {experiment_dir}")

    # Final GPU memory cleanup and stats
    if device.type == 'cuda':
        final_memory_before_cleanup = torch.cuda.memory_allocated(device) / 1024**3
        torch.cuda.empty_cache()
        final_memory_after_cleanup = torch.cuda.memory_allocated(device) / 1024**3
        peak_memory = torch.cuda.max_memory_allocated(device) / 1024**3

        logger.info(f"GPU Memory before cleanup: {final_memory_before_cleanup:.2f}GB")
        logger.info(f"GPU Memory after cleanup: {final_memory_after_cleanup:.2f}GB")
        logger.info(f"Peak GPU Memory usage: {peak_memory:.2f}GB")

    logger.info("="*80)

    # Return success code
    return 0 if len(train_history) > 0 else 1


if __name__ == '__main__':
    try:
        exit_code = main()
        exit(exit_code)
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user (Ctrl+C)")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)