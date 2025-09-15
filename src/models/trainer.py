import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import pandas as pd
import polars as pl
from pathlib import Path
import logging
import time
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False
    wandb = None

try:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    # Fallback implementations
    def mean_absolute_error(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    def mean_squared_error(y_true, y_pred):
        return np.mean((y_true - y_pred) ** 2)
    
    def r2_score(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None
    sns = None

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # Fallback progress bar
    def tqdm(iterable, desc="Progress", **kwargs):
        print(f"{desc}...")
        return iterable

import warnings
warnings.filterwarnings('ignore')

from .patchtst import PatchTST, create_patchtst_model, get_config, get_training_config

logger = logging.getLogger(__name__)

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class FinancialLoss(nn.Module):
    """
    Advanced loss function designed for financial time series analysis.
    Combines directional accuracy with magnitude prediction for better trend analysis.
    """

    def __init__(self, alpha=0.6, beta=0.3, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Weight for directional accuracy
        self.beta = beta    # Weight for magnitude accuracy
        self.gamma = gamma  # Weight for volatility consistency

        self.mse_loss = nn.MSELoss()
        self.huber_loss = nn.HuberLoss(delta=0.1)  # Less sensitive to outliers

    def forward(self, predictions, targets):
        """
        Calculate combined loss for financial analysis:
        - Directional accuracy (trend prediction)
        - Magnitude accuracy (price level prediction)
        - Volatility consistency (realistic movements)
        """
        batch_size = predictions.size(0)
        seq_len = predictions.size(1)

        # Basic magnitude loss (Huber is more robust than MSE for financial data)
        magnitude_loss = self.huber_loss(predictions, targets)

        # Directional accuracy loss
        if seq_len > 1:
            pred_directions = torch.sign(predictions[:, 1:] - predictions[:, :-1])
            true_directions = torch.sign(targets[:, 1:] - targets[:, :-1])
            directional_accuracy = torch.mean((pred_directions == true_directions).float())
            directional_loss = 1.0 - directional_accuracy
        else:
            directional_loss = torch.tensor(0.0, device=predictions.device)

        # Volatility consistency loss (prevent unrealistic jumps)
        if seq_len > 1:
            pred_changes = torch.abs(predictions[:, 1:] - predictions[:, :-1])
            true_changes = torch.abs(targets[:, 1:] - targets[:, :-1])
            volatility_loss = self.mse_loss(pred_changes, true_changes)
        else:
            volatility_loss = torch.tensor(0.0, device=predictions.device)

        # Combine losses with professional weighting
        total_loss = (self.alpha * directional_loss +
                     self.beta * magnitude_loss +
                     self.gamma * volatility_loss)

        return total_loss


class Metrics:
    """Calculate financial analysis metrics"""

    @staticmethod
    def directional_accuracy(predictions, targets):
        """Calculate directional accuracy - crucial for trend analysis"""
        if len(predictions.shape) > 1 and predictions.shape[1] > 1:
            pred_directions = np.sign(predictions[:, 1:] - predictions[:, :-1])
            true_directions = np.sign(targets[:, 1:] - targets[:, :-1])
            return np.mean(pred_directions == true_directions)
        return 0.0

    @staticmethod
    def volatility_similarity(predictions, targets):
        """Measure how well predicted volatility matches actual volatility"""
        pred_vol = np.std(predictions, axis=1) if len(predictions.shape) > 1 else np.std(predictions)
        true_vol = np.std(targets, axis=1) if len(targets.shape) > 1 else np.std(targets)

        # Calculate correlation between predicted and actual volatility
        if isinstance(pred_vol, np.ndarray) and len(pred_vol) > 1:
            correlation = np.corrcoef(pred_vol, true_vol)[0, 1]
            return correlation if not np.isnan(correlation) else 0.0
        return 0.0

    @staticmethod
    def trend_consistency(predictions, targets):
        """Measure consistency of trend predictions over time"""
        if len(predictions.shape) > 1 and predictions.shape[1] > 2:
            # Calculate multi-step directional consistency
            accuracies = []
            for i in range(1, predictions.shape[1]):
                pred_dirs = np.sign(predictions[:, i:] - predictions[:, :-i])
                true_dirs = np.sign(targets[:, i:] - targets[:, :-i])
                acc = np.mean(pred_dirs == true_dirs)
                accuracies.append(acc)
            return np.mean(accuracies)
        return 0.0


@dataclass
class TrainingMetrics:
    """Data class to store training metrics"""
    epoch: int
    train_loss: float
    val_loss: float
    train_mae: float
    val_mae: float
    train_mse: float
    val_mse: float
    train_r2: float
    val_r2: float
    learning_rate: float
    gpu_memory_used: float
    epoch_time: float


class StockDataset(Dataset):
    """
    PyTorch Dataset for stock time series data.
    """
    
    def __init__(
        self,
        data: pl.DataFrame,
        sequence_length: int = 512,
        prediction_length: int = 20,
        target_column: str = "close",
        feature_columns: Optional[List[str]] = None,
        symbol_column: str = "symbol",
        datetime_column: str = "datetime",
        cache_sequences: bool = True,
        normalize: bool = True
    ):
        self.sequence_length = sequence_length
        self.prediction_length = prediction_length
        self.target_column = target_column
        self.symbol_column = symbol_column
        self.datetime_column = datetime_column
        self.cache_sequences = cache_sequences
        self.normalize = normalize
        
        # Auto-detect feature columns if not provided
        if feature_columns is None:
            exclude_cols = {datetime_column, symbol_column, target_column}
            feature_columns = [col for col in data.columns if col not in exclude_cols]
        
        self.feature_columns = feature_columns
        logger.info(f"Dataset initialized with {len(feature_columns)} features")
        
        # Sort data and prepare sequences
        self.data = data.sort([symbol_column, datetime_column]) if symbol_column in data.columns else data.sort(datetime_column)
        self.sequences = self._create_sequences()
        
        logger.info(f"Created {len(self.sequences)} sequences for training")
    
    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create sequences from the time series data"""
        sequences = []
        
        if self.symbol_column in self.data.columns:
            # Multi-symbol dataset
            symbols = self.data[self.symbol_column].unique().to_list()
            
            for symbol in symbols:
                symbol_data = self.data.filter(pl.col(self.symbol_column) == symbol)
                symbol_sequences = self._create_symbol_sequences(symbol_data, symbol)
                sequences.extend(symbol_sequences)
        else:
            # Single symbol dataset
            sequences = self._create_symbol_sequences(self.data, "single")
        
        return sequences
    
    def _create_symbol_sequences(self, data: pl.DataFrame, symbol: str) -> List[Dict[str, Any]]:
        """Create sequences for a single symbol"""
        sequences = []
        data_len = len(data)
        
        if data_len < self.sequence_length + self.prediction_length:
            logger.warning(f"Insufficient data for {symbol}: {data_len} records")
            return sequences
        
        # Extract features and target prices
        features = data.select(self.feature_columns).to_numpy().astype(np.float32)
        target_prices = data[self.target_column].to_numpy().astype(np.float32)
        dates = data[self.datetime_column].to_list()
        
        # Calculate relative returns as target (CRITICAL FIX)
        relative_returns = np.zeros_like(target_prices)
        relative_returns[1:] = (target_prices[1:] - target_prices[:-1]) / (target_prices[:-1] + 1e-8)
        
        # Normalize features if requested
        if self.normalize:
            # Store normalization parameters for inverse transform
            feature_means = np.mean(features, axis=0)
            feature_stds = np.std(features, axis=0) + 1e-8
            features = (features - feature_means) / feature_stds
            
            # Normalize relative returns, not absolute prices
            target_mean = np.mean(relative_returns)
            target_std = np.std(relative_returns) + 1e-8
            normalized_returns = (relative_returns - target_mean) / target_std
        else:
            feature_means = np.zeros(features.shape[1])
            feature_stds = np.ones(features.shape[1])
            target_mean = 0.0
            target_std = 1.0
            normalized_returns = relative_returns
        
        # Create sliding window sequences
        for i in range(data_len - self.sequence_length - self.prediction_length + 1):
            seq_features = features[i:i + self.sequence_length]
            # Use relative returns as target, not absolute prices
            seq_target = normalized_returns[i + self.sequence_length:i + self.sequence_length + self.prediction_length]
            seq_dates = dates[i:i + self.sequence_length]
            # Store last price for conversion back to absolute predictions
            last_price = target_prices[i + self.sequence_length - 1]
            
            sequences.append({
                'features': seq_features,
                'target': seq_target,
                'symbol': symbol,
                'dates': seq_dates,
                'feature_means': feature_means,
                'feature_stds': feature_stds,
                'target_mean': target_mean,
                'target_std': target_std,
                'last_price': last_price  # CRITICAL: Store for prediction conversion
            })
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        seq = self.sequences[idx]
        
        features = torch.from_numpy(seq['features']).float()
        target = torch.from_numpy(seq['target']).float()
        
        metadata = {
            'symbol': seq['symbol'],
            'dates': [d.timestamp() if hasattr(d, 'timestamp') else d for d in seq['dates']],
            'feature_means': seq['feature_means'],
            'feature_stds': seq['feature_stds'],
            'target_mean': seq['target_mean'],
            'target_std': seq['target_std'],
            'last_price': seq.get('last_price', 0.0)  # CRITICAL: Include last_price for conversion
        }
        
        return features, target, metadata


class PatchTSTTrainer:
    
    def __init__(
        self,
        model: PatchTST,
        train_dataset: StockDataset,
        val_dataset: StockDataset,
        config: Dict[str, Any],
        experiment_name: str = "patchtst_stock_prediction",
        save_dir: str = "experiments",
        use_wandb: bool = True
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.experiment_name = experiment_name
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.use_wandb = use_wandb
        
        # Device setup
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        if torch.cuda.is_available():
            logger.info(f"GPU: {torch.cuda.get_device_name()}")
            logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Compile model for PyTorch 2.0 speedup
        if config.get('compile', True) and hasattr(torch, 'compile'):
            logger.info("Compiling model with PyTorch 2.0...")
            self.model = torch.compile(self.model)
        
        # Setup data loaders
        self.train_loader = self._create_dataloader(train_dataset, shuffle=True)
        self.val_loader = self._create_dataloader(val_dataset, shuffle=False)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        self.criterion = self._create_criterion()
        
        # Mixed precision setup
        self.use_amp = config.get('use_amp', True)
        self.scaler = GradScaler() if self.use_amp else None
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.metrics_history = []
        self.early_stopping_counter = 0
        
        # Initialize wandb if enabled
        if self.use_wandb:
            self._init_wandb()
        
        logger.info("Trainer initialized successfully")
    
    def _create_dataloader(self, dataset: StockDataset, shuffle: bool = True) -> DataLoader:
        return DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=shuffle,
            num_workers=self.config.get('num_workers', 16), 
            pin_memory=self.config.get('pin_memory', True),
            persistent_workers=True,
            prefetch_factor=4, 
            drop_last=True if shuffle else False
        )
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer with optimal settings for financial time series"""
        return optim.AdamW(
            self.model.parameters(),
            lr=self.config.get('learning_rate', 1e-4),
            weight_decay=self.config.get('weight_decay', 1e-4),
            betas=self.config.get('betas', (0.9, 0.999)),
            eps=self.config.get('eps', 1e-8)
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler"""
        scheduler_type = self.config.get('scheduler', 'cosine_with_warmup')
        
        if scheduler_type == 'cosine_with_warmup':
            from torch.optim.lr_scheduler import CosineAnnealingLR
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.config.get('max_steps', 100000),
                eta_min=self.config.get('learning_rate', 1e-4) * 0.01
            )
        elif scheduler_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True
            )
        else:
            return None
    
    def _create_criterion(self) -> nn.Module:
        """Create advanced loss function optimized for financial analysis"""
        return FinancialLoss(alpha=0.6, beta=0.3, gamma=0.1)
    
    def _init_wandb(self):
        """Initialize Weights & Biases tracking"""
        if not HAS_WANDB:
            logger.warning("wandb not available, disabling experiment tracking")
            self.use_wandb = False
            return
            
        try:
            wandb.init(
                project="stock-prediction-patchtst",
                name=self.experiment_name,
                config=self.config,
                tags=["patchtst", "stock-prediction"]
            )
            wandb.watch(self.model, log="all", log_freq=1000)
            logger.info("Wandb initialized successfully")
        except Exception as e:
            logger.warning(f"Could not initialize wandb: {e}")
            self.use_wandb = False
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_predictions = []
        epoch_targets = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, (features, targets, metadata) in enumerate(progress_bar):
            # Move data to device
            features = features.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            
            # Forward pass with mixed precision
            if self.use_amp and self.scaler:
                with autocast():
                    outputs = self.model(features)
                    predictions = outputs['predictions']
                    loss = self.criterion(predictions, targets)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    # Gradient clipping
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('grad_clip_norm', 1.0)
                    )
                    
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard precision training
                outputs = self.model(features)
                predictions = outputs['predictions']
                loss = self.criterion(predictions, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), 
                        self.config.get('grad_clip_norm', 1.0)
                    )
                    self.optimizer.step()
                    self.optimizer.zero_grad()
            
            # Store metrics
            epoch_losses.append(loss.item())
            epoch_predictions.append(predictions.detach().cpu().numpy())
            epoch_targets.append(targets.detach().cpu().numpy())
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.6f}',
                'lr': f'{self.optimizer.param_groups[0]["lr"]:.2e}',
                'vram': f'{torch.cuda.memory_allocated() / 1024**3:.1f}GB'
            })
        
        # Calculate epoch metrics
        epoch_predictions = np.concatenate(epoch_predictions, axis=0)
        epoch_targets = np.concatenate(epoch_targets, axis=0)
        
        train_loss = np.mean(epoch_losses)
        train_mae = mean_absolute_error(epoch_targets.flatten(), epoch_predictions.flatten())
        train_mse = mean_squared_error(epoch_targets.flatten(), epoch_predictions.flatten())
        train_r2 = r2_score(epoch_targets.flatten(), epoch_predictions.flatten())

        # Professional financial metrics for trend analysis
        train_directional_acc = Metrics.directional_accuracy(
            epoch_predictions, epoch_targets
        )
        train_volatility_sim = Metrics.volatility_similarity(
            epoch_predictions, epoch_targets
        )
        train_trend_consistency = Metrics.trend_consistency(
            epoch_predictions, epoch_targets
        )

        return {
            'train_loss': train_loss,
            'train_mae': train_mae,
            'train_mse': train_mse,
            'train_r2': train_r2,
            'train_directional_accuracy': train_directional_acc,
            'train_volatility_similarity': train_volatility_sim,
            'train_trend_consistency': train_trend_consistency
        }
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch"""
        # Check if validation dataset is empty
        if len(self.val_dataset) == 0 or len(self.val_loader) == 0:
            logger.warning("No validation data available, skipping validation")
            return {
                'val_loss': 0.0,
                'val_mae': 0.0,
                'val_mse': 0.0,
                'val_r2': 0.0
            }
        
        self.model.eval()
        epoch_losses = []
        epoch_predictions = []
        epoch_targets = []
        
        with torch.no_grad():
            for features, targets, metadata in tqdm(self.val_loader, desc="Validation"):
                features = features.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(features)
                        predictions = outputs['predictions']
                        loss = self.criterion(predictions, targets)
                else:
                    outputs = self.model(features)
                    predictions = outputs['predictions']
                    loss = self.criterion(predictions, targets)
                
                epoch_losses.append(loss.item())
                epoch_predictions.append(predictions.cpu().numpy())
                epoch_targets.append(targets.cpu().numpy())
        
        # Calculate validation metrics
        epoch_predictions = np.concatenate(epoch_predictions, axis=0)
        epoch_targets = np.concatenate(epoch_targets, axis=0)
        
        val_loss = np.mean(epoch_losses)
        val_mae = mean_absolute_error(epoch_targets.flatten(), epoch_predictions.flatten())
        val_mse = mean_squared_error(epoch_targets.flatten(), epoch_predictions.flatten())
        val_r2 = r2_score(epoch_targets.flatten(), epoch_predictions.flatten())

        # Professional financial metrics for validation
        val_directional_acc = Metrics.directional_accuracy(
            epoch_predictions, epoch_targets
        )
        val_volatility_sim = Metrics.volatility_similarity(
            epoch_predictions, epoch_targets
        )
        val_trend_consistency = Metrics.trend_consistency(
            epoch_predictions, epoch_targets
        )

        return {
            'val_loss': val_loss,
            'val_mae': val_mae,
            'val_mse': val_mse,
            'val_r2': val_r2,
            'val_directional_accuracy': val_directional_acc,
            'val_volatility_similarity': val_volatility_sim,
            'val_trend_consistency': val_trend_consistency
        }
    
    def train(self) -> List[TrainingMetrics]:
        """Main training loop """
        logger.info(f"Starting training for {self.config.get('epochs', 100)} epochs")
        
        start_time = time.time()
        
        for epoch in range(self.current_epoch, self.config.get('epochs', 100)):
            self.current_epoch = epoch
            epoch_start_time = time.time()
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Validate epoch
            val_metrics = self.validate_epoch()
            
            # Update scheduler
            if self.scheduler:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['val_loss'])
                else:
                    self.scheduler.step()
            
            # Calculate additional metrics
            epoch_time = time.time() - epoch_start_time
            gpu_memory = torch.cuda.memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Create metrics object
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_metrics['train_loss'],
                val_loss=val_metrics['val_loss'],
                train_mae=train_metrics['train_mae'],
                val_mae=val_metrics['val_mae'],
                train_mse=train_metrics['train_mse'],
                val_mse=val_metrics['val_mse'],
                train_r2=train_metrics['train_r2'],
                val_r2=val_metrics['val_r2'],
                learning_rate=current_lr,
                gpu_memory_used=gpu_memory,
                epoch_time=epoch_time
            )
            
            self.metrics_history.append(metrics)
            
            # Log metrics
            self._log_metrics(metrics)
            
            # Save checkpoint
            if val_metrics['val_loss'] < self.best_val_loss:
                self.best_val_loss = val_metrics['val_loss']
                self.early_stopping_counter = 0
                self._save_checkpoint('best')
            else:
                self.early_stopping_counter += 1
            
            # Regular checkpoint
            if epoch % self.config.get('save_every', 10) == 0:
                self._save_checkpoint(f'epoch_{epoch}')
            
            # Early stopping
            if self.early_stopping_counter >= self.config.get('early_stopping_patience', 20):
                logger.info(f"Early stopping at epoch {epoch}")
                break
            
            # Print epoch summary
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_metrics['train_loss']:.6f}, "
                f"Val Loss: {val_metrics['val_loss']:.6f}, "
                f"Val R2: {val_metrics['val_r2']:.4f}, "
                f"Time: {epoch_time:.2f}s, "
                f"VRAM: {gpu_memory:.1f}GB"
            )
        
        total_time = time.time() - start_time
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
        
        # Save final model
        self._save_checkpoint('final')
        
        return self.metrics_history
    
    def _log_metrics(self, metrics: TrainingMetrics):
        """Log metrics to wandb and local files"""
        if self.use_wandb and HAS_WANDB:
            wandb.log({
                'epoch': metrics.epoch,
                'train_loss': metrics.train_loss,
                'val_loss': metrics.val_loss,
                'train_mae': metrics.train_mae,
                'val_mae': metrics.val_mae,
                'train_r2': metrics.train_r2,
                'val_r2': metrics.val_r2,
                'learning_rate': metrics.learning_rate,
                'gpu_memory_gb': metrics.gpu_memory_used,
                'epoch_time': metrics.epoch_time
            })
        
        # Save metrics to JSON
        metrics_file = self.save_dir / 'metrics.json'
        with open(metrics_file, 'w') as f:
            metrics_data = [m.__dict__ for m in self.metrics_history]
            json.dump(metrics_data, f, indent=2)
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'metrics_history': [m.__dict__ for m in self.metrics_history]
        }
        
        checkpoint_path = self.save_dir / f'{name}_checkpoint.pt'
        torch.save(checkpoint, checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint.get('scheduler_state_dict'):
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        if self.scaler and checkpoint.get('scaler_state_dict'):
            self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        
        logger.info(f"Checkpoint loaded from epoch {self.current_epoch}")
    
    def predict(self, dataset: StockDataset) -> Dict[str, np.ndarray]:
        """Generate predictions on a dataset"""
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_metadata = []
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.get('batch_size', 32),
            shuffle=False,
            num_workers=self.config.get('num_workers', 8),
            pin_memory=True
        )
        
        # Check if dataset is empty
        if len(dataset) == 0:
            logger.warning("Dataset is empty, returning empty predictions")
            return {
                'predictions': np.array([]),
                'targets': np.array([]),
                'metadata': []
            }
        
        with torch.no_grad():
            for features, targets, metadata in tqdm(dataloader, desc="Predicting"):
                features = features.to(self.device)
                
                if self.use_amp:
                    with autocast():
                        outputs = self.model(features)
                        predictions = outputs['predictions']
                else:
                    outputs = self.model(features)
                    predictions = outputs['predictions']
                
                all_predictions.append(predictions.cpu().numpy())
                all_targets.append(targets.numpy())
                all_metadata.extend(metadata)
        
        # Handle case where no predictions were generated
        if not all_predictions:
            logger.warning("No predictions generated, returning empty arrays")
            return {
                'predictions': np.array([]),
                'targets': np.array([]),
                'metadata': all_metadata
            }
        
        return {
            'predictions': np.concatenate(all_predictions, axis=0),
            'targets': np.concatenate(all_targets, axis=0),
            'metadata': all_metadata
        }
    
    def plot_training_history(self):
        """Plot training metrics"""
        if not HAS_MATPLOTLIB:
            logger.warning("matplotlib not available, cannot plot training history")
            return
            
        if not self.metrics_history:
            logger.warning("No metrics history available")
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        
        epochs = [m.epoch for m in self.metrics_history]
        train_losses = [m.train_loss for m in self.metrics_history]
        val_losses = [m.val_loss for m in self.metrics_history]
        train_r2 = [m.train_r2 for m in self.metrics_history]
        val_r2 = [m.val_r2 for m in self.metrics_history]
        learning_rates = [m.learning_rate for m in self.metrics_history]
        gpu_memory = [m.gpu_memory_used for m in self.metrics_history]
        
        # Loss plot
        axes[0, 0].plot(epochs, train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(epochs, val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # R� plot
        axes[0, 1].plot(epochs, train_r2, label='Train R�', color='green')
        axes[0, 1].plot(epochs, val_r2, label='Val R�', color='orange')
        axes[0, 1].set_title('R� Score')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('R�')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Learning rate plot
        axes[0, 2].plot(epochs, learning_rates, color='purple')
        axes[0, 2].set_title('Learning Rate')
        axes[0, 2].set_xlabel('Epoch')
        axes[0, 2].set_ylabel('Learning Rate')
        axes[0, 2].grid(True)
        axes[0, 2].set_yscale('log')
        
        # GPU memory usage
        axes[1, 0].plot(epochs, gpu_memory, color='red')
        axes[1, 0].set_title('GPU Memory Usage')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Memory (GB)')
        axes[1, 0].grid(True)
        
        # MAE comparison
        train_mae = [m.train_mae for m in self.metrics_history]
        val_mae = [m.val_mae for m in self.metrics_history]
        axes[1, 1].plot(epochs, train_mae, label='Train MAE', color='cyan')
        axes[1, 1].plot(epochs, val_mae, label='Val MAE', color='magenta')
        axes[1, 1].set_title('Mean Absolute Error')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('MAE')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Epoch time
        epoch_times = [m.epoch_time for m in self.metrics_history]
        axes[1, 2].plot(epochs, epoch_times, color='brown')
        axes[1, 2].set_title('Epoch Time')
        axes[1, 2].set_xlabel('Epoch')
        axes[1, 2].set_ylabel('Time (seconds)')
        axes[1, 2].grid(True)
        
        plt.tight_layout()
        plot_path = self.save_dir / 'training_history.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Training history plot saved to {plot_path}")


def create_trainer_from_config(
    model_config: Dict[str, Any],
    training_config: Dict[str, Any],
    train_data: pl.DataFrame,
    val_data: pl.DataFrame,
    experiment_name: str = "patchtst"
) -> PatchTSTTrainer:
    """
    Factory function to create trainer from configuration.
    """
    
    # Create model
    model = create_patchtst_model(model_config)
    
    # Create datasets
    train_dataset = StockDataset(
        data=train_data,
        sequence_length=model_config['seq_len'],
        prediction_length=model_config['pred_len'],
        normalize=True,
        cache_sequences=True
    )
    
    val_dataset = StockDataset(
        data=val_data,
        sequence_length=model_config['seq_len'],
        prediction_length=model_config['pred_len'],
        normalize=True,
        cache_sequences=True
    )
    
    # Combine configurations
    config = {**model_config, **training_config}
    
    # Create trainer
    trainer = PatchTSTTrainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        config=config,
        experiment_name=experiment_name,
        use_wandb=True
    )
    
    return trainer


# Example usage
if __name__ == "__main__":
    # Example of how to use the trainer
    print("PatchTST Trainer")
    
    # Get optimized configurations
    model_config = get_config("large")  
    training_config = get_training_config()
    
    print(f"Model configuration: {model_config}")
    print(f"Training configuration: {training_config}")
    
    # This is where you would load your preprocessed data
    # train_data = pl.read_parquet("data/processed/train_with_indicators.parquet")
    # val_data = pl.read_parquet("data/processed/val_with_indicators.parquet")
    
    # trainer = create_trainer_from_config(
    #     model_config=model_config,
    #     training_config=training_config,
    #     train_data=train_data,
    #     val_data=val_data,
    #     experiment_name="patchtst_stock"
    # )
    
    # Start training
    # metrics = trainer.train()
    
    # Plot results
    # trainer.plot_training_history()
    
    logger.info("Trainer implementation complete!")