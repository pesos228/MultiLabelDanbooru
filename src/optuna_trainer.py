"""
Optuna hyperparameter optimization with detailed logging and saving
"""
import optuna
import logging
import json
import sqlite3
from pathlib import Path
from datetime import datetime
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import your existing modules
from dataset import create_train_val_datasets
from models import get_model
from engine import train_one_epoch, evaluate
from utils import set_seed, force_gpu_usage, setup_logging, count_parameters
from main_train import create_amp_scaler

logger = logging.getLogger(__name__)

class OptunaTrainer:
    def __init__(self, args):
        self.args = args
        self.device = force_gpu_usage()
        
        # Создаем датасеты один раз (экономим время)
        logger.info("Creating datasets for Optuna optimization...")
        self.train_dataset, self.val_dataset = create_train_val_datasets(
            csv_path=Path(args.data_csv),
            root_img_dir=Path(args.img_root),
            image_size=args.image_size,
            top_k_tags=args.top_k_tags,
            min_tag_frequency=args.min_tag_frequency,
            min_tags_per_image=args.min_tags_per_image,
            val_size=args.val_size,
            random_state=42
        )
        
        self.num_classes = self.train_dataset.num_classes
        logger.info(f"Dataset ready. Classes: {self.num_classes}")
        
        # Создаем папку для сохранения результатов каждого trial
        self.trials_dir = Path(args.output_dir) / "trials"
        self.trials_dir.mkdir(parents=True, exist_ok=True)
        
    def objective(self, trial):
        """Optuna objective function with detailed logging"""
        trial_number = trial.number
        
        try:
            # ===========================================
            # HYPERPARAMETERS TO OPTIMIZE
            # ===========================================
            
            # Learning rate (log scale)
            lr = trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True)
            
            # Weight decay (log scale) 
            weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
            
            # Batch size (discrete values)
            batch_size = trial.suggest_categorical('batch_size', [16, 24, 32, 48, 64])
            
            # Optimizer
            optimizer_name = trial.suggest_categorical('optimizer_name', ['adamw', 'sgd'])
            
            # Scheduler
            scheduler_name = trial.suggest_categorical('scheduler_name', ['cosine', 'step', 'plateau'])
            
            # ===========================================
            # TRIAL SETUP
            # ===========================================
            
            trial_start_time = datetime.now()
            
            # Создаем папку для этого trial'а
            trial_dir = self.trials_dir / f"trial_{trial_number:03d}"
            trial_dir.mkdir(exist_ok=True)
            
            # Логгер для конкретного trial'а
            trial_log_file = trial_dir / 'trial.log'
            trial_logger = logging.getLogger(f'trial_{trial_number}')
            trial_logger.setLevel(logging.INFO)
            trial_handler = logging.FileHandler(trial_log_file, encoding='utf-8')
            trial_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
            trial_logger.addHandler(trial_handler)
            
            # Сохраняем конфигурацию trial'а
            trial_config = {
                'trial_number': trial_number,
                'hyperparameters': {
                    'learning_rate': lr,
                    'weight_decay': weight_decay,
                    'batch_size': batch_size,
                    'optimizer_name': optimizer_name,
                    'scheduler_name': scheduler_name
                },
                'dataset_info': {
                    'model_name': self.args.model_name,
                    'num_classes': self.num_classes,
                    'train_size': len(self.train_dataset),
                    'val_size': len(self.val_dataset),
                    'top_k_tags': self.args.top_k_tags,
                    'image_size': self.args.image_size,
                    'val_size_ratio': self.args.val_size,
                    'min_tag_frequency': self.args.min_tag_frequency,
                    'min_tags_per_image': self.args.min_tags_per_image
                },
                'training_config': {
                    'epochs': self.args.epochs,
                    'patience': self.args.patience,
                    'use_amp': self.args.use_amp,
                    'num_workers': self.args.num_workers,
                    'seed': self.args.seed
                },
                'start_time': trial_start_time.isoformat(),
                'device': str(self.device),
                'pytorch_version': torch.__version__
            }
            
            # Сохраняем конфиг
            with open(trial_dir / 'config.json', 'w') as f:
                json.dump(trial_config, f, indent=2)
            
            logger.info("="*80)
            logger.info(f"OPTUNA TRIAL #{trial_number}")
            logger.info(f"Learning Rate: {lr:.2e}")
            logger.info(f"Weight Decay: {weight_decay:.2e}")  
            logger.info(f"Batch Size: {batch_size}")
            logger.info(f"Optimizer: {optimizer_name}")
            logger.info(f"Scheduler: {scheduler_name}")
            logger.info(f"Trial directory: {trial_dir}")
            logger.info("="*80)
            
            trial_logger.info("="*80)
            trial_logger.info(f"STARTING TRIAL #{trial_number}")
            trial_logger.info(f"Parameters: {trial_config['hyperparameters']}")
            trial_logger.info("="*80)
            
            # ===========================================
            # CREATE MODEL & TRAINING COMPONENTS
            # ===========================================
            
            # DataLoaders
            train_loader = DataLoader(
                self.train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
                persistent_workers=True if self.args.num_workers > 0 else False
            )
            
            val_loader = DataLoader(
                self.val_dataset,  
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.num_workers,
                pin_memory=True,
                persistent_workers=True if self.args.num_workers > 0 else False
            )
            
            # Model
            model = get_model(self.args.model_name, self.num_classes)
            model = model.to(self.device)
            
            # Логируем информацию о модели
            total_params = count_parameters(model)
            trial_logger.info(f"Model: {self.args.model_name}")
            trial_logger.info(f"Total parameters: {total_params:,}")
            trial_logger.info(f"Model device: {next(model.parameters()).device}")
            
            # Loss function
            criterion = nn.BCEWithLogitsLoss()
            
            # Optimizer based on trial suggestion
            if optimizer_name == 'adamw':
                optimizer = torch.optim.AdamW(
                    model.parameters(),
                    lr=lr,
                    weight_decay=weight_decay
                )
                optimizer_info = f"AdamW(lr={lr:.2e}, weight_decay={weight_decay:.2e})"
            else:  # sgd
                optimizer = torch.optim.SGD(
                    model.parameters(),
                    lr=lr,
                    momentum=0.9,
                    weight_decay=weight_decay
                )
                optimizer_info = f"SGD(lr={lr:.2e}, momentum=0.9, weight_decay={weight_decay:.2e})"
            
            trial_logger.info(f"Optimizer: {optimizer_info}")
            
            # Scheduler based on trial suggestion
            if scheduler_name == 'cosine':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer,
                    T_max=self.args.epochs * len(train_loader),
                    eta_min=lr * 0.01
                )
                scheduler_info = f"CosineAnnealingLR(T_max={self.args.epochs * len(train_loader)}, eta_min={lr * 0.01:.2e})"
            elif scheduler_name == 'step':
                step_size = max(1, self.args.epochs // 3)
                scheduler = torch.optim.lr_scheduler.StepLR(
                    optimizer,
                    step_size=step_size,
                    gamma=0.1
                )
                scheduler_info = f"StepLR(step_size={step_size}, gamma=0.1)"
            elif scheduler_name == 'plateau':
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                    optimizer,
                    mode='max',
                    factor=0.5,
                    patience=2,
                    verbose=True
                )
                scheduler_info = "ReduceLROnPlateau(mode=max, factor=0.5, patience=2)"
            
            trial_logger.info(f"Scheduler: {scheduler_info}")
            
            # AMP Scaler
            scaler = create_amp_scaler(self.args.use_amp, self.device)
            if scaler is not None:
                trial_logger.info("Using Automatic Mixed Precision (AMP)")
            else:
                trial_logger.info("Using FP32 training")
            
            # ===========================================
            # TRAINING LOOP WITH DETAILED LOGGING
            # ===========================================
            
            best_macro_f1 = 0.0
            best_epoch = 0
            patience_counter = 0
            
            # История обучения для этого trial'а
            train_history = []
            val_history = []
            
            trial_logger.info("Starting training loop...")
            trial_logger.info("="*80)
            
            for epoch in range(1, self.args.epochs + 1):
                epoch_start_time = datetime.now()
                
                # Training
                train_metrics = train_one_epoch(
                    model=model,
                    dataloader=train_loader,
                    criterion=criterion,
                    optimizer=optimizer,
                    device=self.device,
                    scaler=scaler,
                    scheduler=scheduler if scheduler_name != 'plateau' else None,
                    accumulation_steps=1,
                    epoch=epoch
                )
                
                # Validation  
                val_metrics = evaluate(
                    model=model,
                    dataloader=val_loader,
                    criterion=criterion,
                    device=self.device,
                    num_classes=self.num_classes,
                    tag_vocab=self.train_dataset.tag_vocab,
                    epoch=epoch
                )
                
                # Scheduler step for plateau
                if scheduler_name == 'plateau':
                    scheduler.step(val_metrics['macro_f1'])
                
                current_f1 = val_metrics['macro_f1']
                epoch_end_time = datetime.now()
                epoch_duration = epoch_end_time - epoch_start_time
                
                # Update best score
                if current_f1 > best_macro_f1:
                    best_macro_f1 = current_f1
                    best_epoch = epoch
                    patience_counter = 0
                    
                    # Сохранить лучшую модель для этого trial'а
                    best_model_path = trial_dir / 'best_model.pth'
                    checkpoint = {
                        'epoch': epoch,
                        'trial_number': trial_number,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                        'scaler_state_dict': scaler.state_dict() if scaler else None,
                        'best_macro_f1': best_macro_f1,
                        'hyperparameters': trial_config['hyperparameters'],
                        'val_metrics': val_metrics,
                        'train_metrics': train_metrics,
                        'tag_vocab': self.train_dataset.tag_vocab,
                        'pytorch_version': torch.__version__
                    }
                    torch.save(checkpoint, best_model_path)
                    
                else:
                    patience_counter += 1
                
                # Сохранить метрики в историю
                train_entry = train_metrics.copy()
                train_entry.update({
                    'epoch': epoch,
                    'trial_number': trial_number,
                    'timestamp': epoch_start_time.isoformat(),
                    'duration_seconds': epoch_duration.total_seconds(),
                    'type': 'train'
                })
                train_history.append(train_entry)
                
                val_entry = val_metrics.copy()
                val_entry.update({
                    'epoch': epoch,
                    'trial_number': trial_number,
                    'timestamp': epoch_start_time.isoformat(),
                    'duration_seconds': epoch_duration.total_seconds(),
                    'type': 'validation',
                    'is_best': current_f1 == best_macro_f1,
                    'patience_counter': patience_counter
                })
                val_history.append(val_entry)
                
                # Report intermediate value to Optuna (for pruning)
                trial.report(current_f1, epoch)
                
                # Детальное логирование каждой эпохи
                trial_logger.info(f"\nEpoch {epoch}/{self.args.epochs} ({epoch_duration})")
                trial_logger.info(f"TRAIN | Loss: {train_metrics['loss']:.4f} | LR: {train_metrics['lr']:.2e}")
                trial_logger.info(f"VAL   | Loss: {val_metrics['loss']:.4f} | Micro F1: {val_metrics['micro_f1']:.4f} | "
                                f"Macro F1: {val_metrics['macro_f1']:.4f} | Hamming: {val_metrics['hamming_loss']:.4f}")
                trial_logger.info(f"BEST  | Macro F1: {best_macro_f1:.4f} (Epoch {best_epoch}) | Patience: {patience_counter}")
                
                # Main logger для прогресса
                if epoch % 2 == 0:
                    logger.info(f"Trial #{trial_number}, Epoch {epoch}: "
                              f"F1={current_f1:.4f}, Best={best_macro_f1:.4f}, Patience={patience_counter}")
                
                # GPU memory info
                if self.device.type == 'cuda' and epoch % 3 == 0:
                    current_memory = torch.cuda.memory_allocated(self.device) / 1024**3
                    max_memory = torch.cuda.max_memory_allocated(self.device) / 1024**3
                    trial_logger.info(f"GPU Memory: {current_memory:.2f}GB (peak: {max_memory:.2f}GB)")
                
                # Pruning check
                if trial.should_prune():
                    trial_logger.info(f"Trial #{trial_number} pruned at epoch {epoch}")
                    logger.info(f"Trial #{trial_number} pruned at epoch {epoch}")
                    raise optuna.TrialPruned()
                
                # Early stopping
                if patience_counter >= self.args.patience:
                    trial_logger.info(f"Early stopping at epoch {epoch} (patience reached)")
                    logger.info(f"Trial #{trial_number}: Early stopping at epoch {epoch}")
                    break
                
                # Сохраняем промежуточную историю каждые 2 эпохи
                if epoch % 2 == 0:
                    interim_history = {
                        'trial_number': trial_number,
                        'current_epoch': epoch,
                        'total_epochs': self.args.epochs,
                        'best_epoch': best_epoch,
                        'best_macro_f1': best_macro_f1,
                        'train_history': train_history,
                        'val_history': val_history,
                        'hyperparameters': trial_config['hyperparameters'],
                        'status': 'running'
                    }
                    
                    with open(trial_dir / 'history_interim.json', 'w') as f:
                        # Convert numpy types to regular for JSON
                        def convert_numpy(obj):
                            if hasattr(obj, 'item'):
                                return obj.item()
                            elif isinstance(obj, dict):
                                return {key: convert_numpy(value) for key, value in obj.items()}
                            elif isinstance(obj, list):
                                return [convert_numpy(item) for item in obj]
                            return obj
                        
                        history_json = convert_numpy(interim_history)
                        json.dump(history_json, f, indent=2, default=str)
            
            # ===========================================
            # FINALIZE TRIAL
            # ===========================================
            
            trial_end_time = datetime.now()
            trial_duration = trial_end_time - trial_start_time
            
            # Clean up GPU memory
            del model, optimizer, scheduler, scaler
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            # Сохранить итоговую историю trial'а
            final_history = {
                'trial_number': trial_number,
                'completed_epochs': len(train_history),
                'total_planned_epochs': self.args.epochs,
                'best_epoch': best_epoch,
                'best_macro_f1': best_macro_f1,
                'final_macro_f1': current_f1,
                'train_history': train_history,
                'val_history': val_history,
                'hyperparameters': trial_config['hyperparameters'],
                'trial_info': {
                    'start_time': trial_start_time.isoformat(),
                    'end_time': trial_end_time.isoformat(),
                    'duration_seconds': trial_duration.total_seconds(),
                    'duration_human': str(trial_duration),
                    'early_stopped': len(train_history) < self.args.epochs,
                    'pruned': False
                },
                'dataset_info': trial_config['dataset_info'],
                'training_config': trial_config['training_config'],
                'status': 'completed'
            }
            
            # Сохранить финальную историю
            with open(trial_dir / 'history.json', 'w') as f:
                def convert_numpy(obj):
                    if hasattr(obj, 'item'):
                        return obj.item()
                    elif isinstance(obj, dict):
                        return {key: convert_numpy(value) for key, value in obj.items()}
                    elif isinstance(obj, list):
                        return [convert_numpy(item) for item in obj]
                    return obj
                
                history_json = convert_numpy(final_history)
                json.dump(history_json, f, indent=2, default=str)
            
            # Сохранить итоговые результаты trial'а
            trial_results = {
                'trial_number': trial_number,
                'best_macro_f1': best_macro_f1,
                'best_epoch': best_epoch,
                'hyperparameters': trial_config['hyperparameters'],
                'duration_seconds': trial_duration.total_seconds(),
                'completed_epochs': len(train_history),
                'status': 'completed'
            }
            
            with open(trial_dir / 'results.json', 'w') as f:
                json.dump(trial_results, f, indent=2)
            
            # Финальные логи
            trial_logger.info("="*80)
            trial_logger.info(f"TRIAL #{trial_number} COMPLETED")
            trial_logger.info(f"Best Macro F1: {best_macro_f1:.4f} (Epoch {best_epoch})")
            trial_logger.info(f"Duration: {trial_duration}")
            trial_logger.info(f"Epochs completed: {len(train_history)}/{self.args.epochs}")
            trial_logger.info(f"Results saved to: {trial_dir}")
            trial_logger.info("="*80)
            
            # Убираем handler чтобы не мешал другим trial'ам
            trial_logger.removeHandler(trial_handler)
            trial_handler.close()
            
            logger.info(f"Trial #{trial_number} completed. Final F1: {best_macro_f1:.4f} (Duration: {trial_duration})")
            
            return best_macro_f1
            
        except optuna.TrialPruned:
            # Обработка pruned trial'а
            trial_end_time = datetime.now()
            
            # Сохраняем информацию о pruned trial'е
            if 'trial_start_time' in locals():
                pruned_info = {
                    'trial_number': trial_number,
                    'status': 'pruned',
                    'last_epoch': len(train_history) if 'train_history' in locals() else 0,
                    'last_f1': train_history[-1]['macro_f1'] if 'train_history' in locals() and train_history else 0.0,
                    'hyperparameters': trial_config['hyperparameters'] if 'trial_config' in locals() else {},
                    'pruned_at': trial_end_time.isoformat(),
                    'duration_seconds': (trial_end_time - trial_start_time).total_seconds()
                }
                
                if 'trial_dir' in locals():
                    with open(trial_dir / 'pruned_info.json', 'w') as f:
                        json.dump(pruned_info, f, indent=2)
                    
                    if 'trial_logger' in locals():
                        trial_logger.info(f"Trial #{trial_number} was pruned")
                        trial_logger.removeHandler(trial_handler)
                        trial_handler.close()
            
            # Clean up GPU memory
            if 'model' in locals():
                del model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            raise
            
        except Exception as e:
            # Обработка ошибок
            trial_end_time = datetime.now()
            
            logger.error(f"Trial #{trial_number} failed: {e}")
            
            # Сохраняем информацию об ошибке
            if 'trial_start_time' in locals():
                error_info = {
                    'trial_number': trial_number,
                    'status': 'failed',
                    'error_message': str(e),
                    'error_type': type(e).__name__,
                    'failed_at': trial_end_time.isoformat(),
                    'hyperparameters': trial_config['hyperparameters'] if 'trial_config' in locals() else {},
                    'duration_seconds': (trial_end_time - trial_start_time).total_seconds()
                }
                
                if 'trial_dir' in locals():
                    with open(trial_dir / 'error_info.json', 'w') as f:
                        json.dump(error_info, f, indent=2)
                    
                    if 'trial_logger' in locals():
                        trial_logger.error(f"Trial #{trial_number} failed: {e}")
                        import traceback
                        trial_logger.error(traceback.format_exc())
                        trial_logger.removeHandler(trial_handler)
                        trial_handler.close()
            
            # Clean up GPU memory
            if 'model' in locals():
                del model
            if self.device.type == 'cuda':
                torch.cuda.empty_cache()
            
            return 0.0


def get_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Optuna hyperparameter optimization')
    
    # Data
    parser.add_argument('--data_csv', type=str, required=True,
                        help='Path to CSV file with metadata')
    parser.add_argument('--img_root', type=str, required=True,
                        help='Path to root directory with images')
    
    # Model  
    parser.add_argument('--model_name', type=str, default='efficientnet_s',
                        help='Model architecture name')
    
    # Dataset parameters
    parser.add_argument('--top_k_tags', type=int, default=500,
                        help='Number of most frequent tags')
    parser.add_argument('--min_tags_per_image', type=int, default=1,
                        help='Minimum tags per image')
    parser.add_argument('--min_tag_frequency', type=int, default=5,
                        help='Minimum tag frequency')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation split ratio')
    
    # Training parameters
    parser.add_argument('--image_size', type=int, default=224,
                        help='Image size')
    parser.add_argument('--epochs', type=int, default=8,
                        help='Number of epochs per trial')
    parser.add_argument('--patience', type=int, default=3,
                        help='Early stopping patience')
    
    # System
    parser.add_argument('--num_workers', type=int, default=4,
                        help='DataLoader workers')
    parser.add_argument('--use_amp', action='store_true', default=True,
                        help='Use automatic mixed precision')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    # Optuna settings
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of optimization trials')
    parser.add_argument('--study_name', type=str, default=None,
                        help='Custom study name')
    
    # Output directory
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save all results')
    
    return parser.parse_args()


def main():
    args = get_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Create output directory structure
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = output_dir / f'optuna_main_{timestamp}.log'
    setup_logging(log_file)
    
    logger.info("="*80)
    logger.info("OPTUNA HYPERPARAMETER OPTIMIZATION")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Trials: {args.n_trials}")
    logger.info(f"Epochs per trial: {args.epochs}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Data CSV: {args.data_csv}")
    logger.info(f"Images root: {args.img_root}")
    logger.info("="*80)
    
    # Determine study name
    if args.study_name is None:
        study_name = f"{args.model_name}_optuna_{timestamp}"
    else:
        study_name = f"{args.study_name}_{timestamp}"
    
    # Save experiment configuration
    experiment_config = {
        'study_name': study_name,
        'timestamp': timestamp,
        'args': vars(args),
        'pytorch_version': torch.__version__,
        'start_time': datetime.now().isoformat()
    }
    
    with open(output_dir / 'experiment_config.json', 'w') as f:
        json.dump(experiment_config, f, indent=2)
    
    # Database path for persistence
    db_path = output_dir / f"{study_name}.db"
    
    logger.info(f"Study name: {study_name}")
    logger.info(f"Database: {db_path}")
    
    # Create Optuna study
    study = optuna.create_study(
        study_name=study_name,
        storage=f"sqlite:///{db_path}",
        direction='maximize',
        load_if_exists=True,
        pruner=optuna.pruners.HyperbandPruner(
            min_resource=3,
            max_resource=args.epochs,
            reduction_factor=2
        ),
        sampler=optuna.samplers.TPESampler(seed=args.seed)
    )
    
    logger.info("Created Optuna study with HyperbandPruner + TPESampler")
    
    # Create trainer
    trainer = OptunaTrainer(args)
    
    # Start optimization
    optimization_start_time = datetime.now()
    
    try:
        logger.info(f"Starting optimization with {args.n_trials} trials...")
        study.optimize(
            trainer.objective,
            n_trials=args.n_trials,
            show_progress_bar=True,
            callbacks=[
                # Сохранение промежуточных результатов каждые 5 trials
                lambda study, trial: save_intermediate_results(study, output_dir) if trial.number % 5 == 0 else None
            ]
        )
        
        optimization_end_time = datetime.now()
        optimization_duration = optimization_end_time - optimization_start_time
        
        logger.info("="*80)
        logger.info("OPTIMIZATION COMPLETED!")
        logger.info(f"Total duration: {optimization_duration}")
        logger.info("="*80)
        
        # ===========================================
        # SAVE COMPREHENSIVE RESULTS
        # ===========================================
        
        # Best results
        best_params = study.best_params
        best_value = study.best_value
        best_trial = study.best_trial
        
        logger.info(f"Best Macro F1: {best_value:.4f}")
        logger.info(f"Best trial: #{best_trial.number}")
        logger.info("Best parameters:")
        for key, value in best_params.items():
            logger.info(f"  {key}: {value}")
        
        # Comprehensive results summary
        comprehensive_results = {
            'experiment_info': {
                'study_name': study_name,
                'timestamp': timestamp,
                'start_time': optimization_start_time.isoformat(),
                'end_time': optimization_end_time.isoformat(),
                'duration_seconds': optimization_duration.total_seconds(),
                'duration_human': str(optimization_duration)
            },
            'optimization_config': {
                'n_trials_planned': args.n_trials,
                'n_trials_completed': len(study.trials),
                'model_name': args.model_name,
                'epochs_per_trial': args.epochs,
                'dataset_config': {
                    'top_k_tags': args.top_k_tags,
                    'image_size': args.image_size,
                    'val_size': args.val_size,
                    'min_tag_frequency': args.min_tag_frequency
                }
            },
            'best_results': {
                'best_macro_f1': float(best_value),
                'best_trial_number': best_trial.number,
                'best_parameters': best_params,
                'best_trial_duration_seconds': best_trial.duration.total_seconds() if best_trial.duration else None
            },
            'trial_statistics': {
                'total_trials': len(study.trials),
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'pruned_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
                'failed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]),
                'running_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.RUNNING])
            },
            'performance_summary': {
                'top_5_trials': [
                    {
                        'trial_number': t.number,
                        'value': t.value,
                        'params': t.params,
                        'duration_seconds': t.duration.total_seconds() if t.duration else None
                    }
                    for t in sorted(study.trials, key=lambda x: x.value or -1, reverse=True)[:5]
                    if t.value is not None
                ],
                'optimization_progress': [
                    {
                        'trial_number': t.number,
                        'value': t.value,
                        'best_so_far': max([trial.value for trial in study.trials[:i+1] if trial.value is not None], default=0)
                    }
                    for i, t in enumerate(study.trials)
                    if t.value is not None
                ]
            }
        }
        
        # Save comprehensive results
        comprehensive_file = output_dir / f"{study_name}_comprehensive_results.json"
        with open(comprehensive_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)
        
        # Save all trial details
        all_trials_details = []
        for trial in study.trials:
            trial_detail = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': str(trial.state),
                'start_time': trial.datetime_start.isoformat() if trial.datetime_start else None,
                'end_time': trial.datetime_complete.isoformat() if trial.datetime_complete else None,
                'duration_seconds': trial.duration.total_seconds() if trial.duration else None,
                'intermediate_values': trial.intermediate_values,
                'user_attrs': trial.user_attrs,
                'system_attrs': trial.system_attrs
            }
            all_trials_details.append(trial_detail)
        
        trials_file = output_dir / f"{study_name}_all_trials.json"
        with open(trials_file, 'w') as f:
            json.dump(all_trials_details, f, indent=2, default=str)
        
        # Save best hyperparameters in easy-to-use format
        best_params_file = output_dir / f"{study_name}_best_hyperparameters.json" 
        best_params_formatted = {
            'best_macro_f1': float(best_value),
            'best_trial_number': best_trial.number,
            'hyperparameters_for_final_training': {
                'learning_rate': best_params['learning_rate'],
                'weight_decay': best_params['weight_decay'],
                'batch_size': best_params['batch_size'],
                'optimizer_name': best_params['optimizer_name'],
                'scheduler_name': best_params['scheduler_name']
            },
            'command_line_args': f"--lr {best_params['learning_rate']:.2e} --weight_decay {best_params['weight_decay']:.2e} --batch_size {best_params['batch_size']} --optimizer_name {best_params['optimizer_name']} --scheduler_name {best_params['scheduler_name']}",
            'notes': f"Best hyperparameters found after {len(study.trials)} trials with Macro F1 = {best_value:.4f}"
        }
        
        with open(best_params_file, 'w') as f:
            json.dump(best_params_formatted, f, indent=2)
        
        # Final logging
        logger.info(f"Results saved to {output_dir}")
        logger.info(f"- Comprehensive results: {comprehensive_file}")
        logger.info(f"- All trials details: {trials_file}")
        logger.info(f"- Best hyperparameters: {best_params_file}")
        logger.info(f"- Individual trial results: {trainer.trials_dir}")
        
        # Print statistics
        completed = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
        pruned = len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])
        failed = len([t for t in study.trials if t.state == optuna.trial.TrialState.FAIL])
        
        logger.info("="*80)
        logger.info("FINAL STATISTICS:")
        logger.info(f"  Completed trials: {completed}")
        logger.info(f"  Pruned trials: {pruned}")  
        logger.info(f"  Failed trials: {failed}")
        logger.info(f"  Total trials: {len(study.trials)}")
        logger.info(f"  Success rate: {completed/len(study.trials)*100:.1f}%")
        
        # Print best hyperparameters for copy-paste
        logger.info("="*80)
        logger.info("BEST HYPERPARAMETERS (COPY-PASTE FOR FINAL TRAINING):")
        logger.info("="*80)
        logger.info(f"--lr {best_params['learning_rate']:.2e}")
        logger.info(f"--weight_decay {best_params['weight_decay']:.2e}")
        logger.info(f"--batch_size {best_params['batch_size']}")
        logger.info(f"--optimizer_name {best_params['optimizer_name']}")
        logger.info(f"--scheduler_name {best_params['scheduler_name']}")
        logger.info("="*80)
        
        return 0
        
    except KeyboardInterrupt:
        optimization_end_time = datetime.now()
        
        logger.info("\nOptimization interrupted by user (Ctrl+C)")
        logger.info("Saving current results...")
        
        if len(study.trials) > 0:
            # Save partial results
            partial_results = {
                'study_name': study_name,
                'interrupted': True,
                'interruption_time': optimization_end_time.isoformat(),
                'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
                'total_trials': len(study.trials),
                'best_value': float(study.best_value) if study.best_value else 0.0,
                'best_params': study.best_params if study.best_params else {},
                'partial_duration_seconds': (optimization_end_time - optimization_start_time).total_seconds()
            }
            
            partial_file = output_dir / f"{study_name}_partial_results.json" 
            with open(partial_file, 'w') as f:
                json.dump(partial_results, f, indent=2)
            
            logger.info(f"Partial results saved to {partial_file}")
            if study.best_value:
                logger.info(f"Best F1 so far: {study.best_value:.4f}")
                logger.info(f"Best params: {study.best_params}")
        
        return 1
        
    except Exception as e:
        logger.error(f"Optimization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def save_intermediate_results(study, output_dir):
    """Save intermediate results during optimization"""
    if len(study.trials) == 0:
        return
    
    intermediate_results = {
        'current_trials': len(study.trials),
        'completed_trials': len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
        'current_best_value': study.best_value if study.best_value else 0.0,
        'current_best_params': study.best_params if study.best_params else {},
        'timestamp': datetime.now().isoformat()
    }
    
    intermediate_file = output_dir / 'intermediate_results.json'
    with open(intermediate_file, 'w') as f:
        json.dump(intermediate_results, f, indent=2)


if __name__ == '__main__':
    exit_code = main()
    exit(exit_code)