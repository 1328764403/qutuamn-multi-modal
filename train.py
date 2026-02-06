"""
Training script for multimodal fusion models
"""

import random
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import yaml
import argparse
import os
from tqdm import tqdm
import json
from pathlib import Path

from models import TFN, LMF, MFN, MulT, GCNFusion, HypergraphFusion, QuantumHybridModel
from utils.data_loader import generate_synthetic_data, get_dataloader
from utils.metrics import calculate_metrics, MetricsTracker
from utils.load_finmme import load_finmme_data
from utils.load_finmultitime import load_finmultitime_data
from utils.load_fcmr import load_fcmr_data
from utils.fcmr_task_switcher import get_fcmr_task_info
from utils.path_guard import resolve_inside_project
from utils.load_a_share import load_a_share_data


def set_global_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Best-effort determinism
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, task_type='regression', is_multilabel=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []   # store batch outputs
    all_labels = []
    
    for modalities, labels in dataloader:
        modalities = [mod.to(device) for mod in modalities]
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(*modalities)
        
        # Loss computation
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    num_batches = len(all_preds)
    if num_batches == 0:
        avg_loss = 0.0
        metrics = {
            'R2': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'MAPE': 0.0
        }
    else:
        avg_loss = total_loss / num_batches
        all_preds_np = np.concatenate(all_preds, axis=0).astype(np.float32)
        all_labels_np = np.concatenate(all_labels, axis=0).astype(np.float32)

        if task_type == 'classification':
            if is_multilabel:
                # Multi-label: sigmoid
                all_preds_np = 1.0 / (1.0 + np.exp(-np.clip(all_preds_np, -50, 50)))
            else:
                # Single-label: softmax
                x = all_preds_np - np.max(all_preds_np, axis=1, keepdims=True)
                exp_x = np.exp(np.clip(x, -50, 50))
                all_preds_np = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)

        verbose = False  # Set to True for debugging
        metrics = calculate_metrics(all_labels_np, all_preds_np, task_type=task_type, verbose=verbose, is_multilabel=is_multilabel)
    
    return avg_loss, metrics


def validate(model, dataloader, criterion, device, task_type='regression', is_multilabel=False):
    """Validate model"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    # Check if dataloader is empty
    if len(dataloader) == 0:
        # Return default metrics for empty validation set
        return 0.0, {
            'R2': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'MAPE': 0.0
        }
    
    with torch.no_grad():
        for modalities, labels in dataloader:
            modalities = [mod.to(device) for mod in modalities]
            labels = labels.to(device)
            
            outputs = model(*modalities)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_labels.append(labels.detach().cpu().numpy())
    
    # Use number of batches instead of len(dataloader) to avoid division by zero
    num_batches = len(all_preds)
    if num_batches == 0:
        avg_loss = 0.0
        metrics = {
            'R2': 0.0,
            'RMSE': 0.0,
            'MAE': 0.0,
            'MSE': 0.0,
            'MAPE': 0.0
        }
    else:
        avg_loss = total_loss / num_batches
        all_preds_np = np.concatenate(all_preds, axis=0).astype(np.float32)
        all_labels_np = np.concatenate(all_labels, axis=0).astype(np.float32)

        if task_type == 'classification':
            if is_multilabel:
                all_preds_np = 1.0 / (1.0 + np.exp(-np.clip(all_preds_np, -50, 50)))
            else:
                x = all_preds_np - np.max(all_preds_np, axis=1, keepdims=True)
                exp_x = np.exp(np.clip(x, -50, 50))
                all_preds_np = exp_x / (np.sum(exp_x, axis=1, keepdims=True) + 1e-12)

        verbose = False  # Set to True for debugging
        metrics = calculate_metrics(all_labels_np, all_preds_np, task_type=task_type, verbose=verbose, is_multilabel=is_multilabel)
    
    return avg_loss, metrics


def initialize_model_weights(model):
    """Initialize model weights using Xavier/Kaiming initialization"""
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # Use Xavier initialization for linear layers
            init.xavier_uniform_(module.weight)
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Conv1d) or isinstance(module, nn.Conv2d):
            # Use Kaiming initialization for conv layers
            init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
            if module.bias is not None:
                init.constant_(module.bias, 0.0)
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            init.constant_(module.weight, 1.0)
            init.constant_(module.bias, 0.0)


def train_model(model_name, model, train_loader, val_loader, config, device, task_type='regression', is_multilabel=False):
    """Train a single model"""
    print(f"\n{'='*50}")
    print(f"Training {model_name}")
    print(f"{'='*50}")
    
    # Initialize model weights properly
    initialize_model_weights(model)
    
    # Loss function
    if task_type == 'classification':
        criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
    else:
        criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Add learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, 
        verbose=True, min_lr=1e-6
    )
    
    tracker = MetricsTracker()
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    
    epochs = config['training']['epochs']
    patience = config['training']['early_stopping_patience']
    
    for epoch in range(epochs):
        # Train
        train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, task_type=task_type, is_multilabel=is_multilabel)
        
        # Validate
        val_loss, val_metrics = validate(model, val_loader, criterion, device, task_type=task_type, is_multilabel=is_multilabel)
        
        # Update tracker
        tracker.update(train_loss, val_loss, train_metrics, val_metrics)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} (LR: {current_lr:.6f})")
            print(f"  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            if task_type == 'classification':
                main_metric = 'F1_Micro' if 'F1_Micro' in val_metrics else list(val_metrics.keys())[0]
                print(f"  Train {main_metric}: {train_metrics.get(main_metric, 0):.4f}, Val {main_metric}: {val_metrics.get(main_metric, 0):.4f}")
            else:
                print(f"  Train R2: {train_metrics['R2']:.4f}, Val R2: {val_metrics['R2']:.4f}")
            
            # Print diagnostic info if R² is negative (regression only)
            if task_type != 'classification' and val_metrics['R2'] < 0 and epoch == 0:
                print(f"  ⚠️  Warning: Negative R² detected. This may be normal in early training.")
                print(f"  Diagnostic: Val RMSE={val_metrics['RMSE']:.4f}, Val MAE={val_metrics['MAE']:.4f}")
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break
    
    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    # Final validation
    final_val_loss, final_val_metrics = validate(model, val_loader, criterion, device, task_type=task_type, is_multilabel=is_multilabel)
    
    print(f"\nFinal Results for {model_name}:")
    print(f"  Val Loss: {final_val_loss:.4f}")
    if task_type == 'classification':
        main_metric = 'F1_Micro' if 'F1_Micro' in final_val_metrics else list(final_val_metrics.keys())[0]
        print(f"  Val {main_metric}: {final_val_metrics.get(main_metric, 0):.4f}")
        if 'Accuracy' in final_val_metrics:
            print(f"  Val Accuracy: {final_val_metrics['Accuracy']:.4f}")
    else:
        print(f"  Val R2: {final_val_metrics['R2']:.4f}")
        print(f"  Val RMSE: {final_val_metrics['RMSE']:.4f}")
    
    return model, tracker, final_val_metrics


def main():
    parser = argparse.ArgumentParser(description='Train multimodal fusion models')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--save_dir', type=str, default=None,
                       help='Override training.save_dir in config')
    parser.add_argument('--n_qubits', type=int, default=None,
                       help='Override model.quantum.n_qubits in config (for sweeps)')
    parser.add_argument('--n_quantum_layers', type=int, default=None,
                       help='Override model.quantum.n_quantum_layers in config (for sweeps)')
    args = parser.parse_args()
    
    # Load config
    config_path = Path(args.config)
    if not config_path.exists() and not config_path.is_absolute():
        # Allow running from repo root: resolve config relative to this script
        alt = (Path(__file__).resolve().parent / config_path).resolve()
        if alt.exists():
            config_path = alt

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # Seed
    set_global_seed(args.seed)
    print(f"Seed: {args.seed}")

    # Optional save dir override
    if args.save_dir:
        config.setdefault('training', {})
        config['training']['save_dir'] = args.save_dir

    # Optional quantum hyperparameter overrides (useful for systematic sweeps)
    if 'model' in config and 'quantum' in config['model']:
        if args.n_qubits is not None:
            print(f"Overriding config: model.quantum.n_qubits = {args.n_qubits}")
            config['model']['quantum']['n_qubits'] = int(args.n_qubits)
        if args.n_quantum_layers is not None:
            print(f"Overriding config: model.quantum.n_quantum_layers = {args.n_quantum_layers}")
            config['model']['quantum']['n_quantum_layers'] = int(args.n_quantum_layers)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Default task settings
    task_type = 'regression'
    is_multilabel = False
    
    # Load data
    data_source = config['data'].get('source', 'synthetic')  # 'synthetic', 'finmme', 'finmultitime', 'fcmr', or 'a_share'
    
    # Print data source info
    print(f"\nData source: {data_source}")
    
    if data_source == 'finmme':
        print("Loading FinMME dataset...")
        max_samples = config['data'].get('max_samples', None)
        if max_samples:
            print(f"快速测试模式: 限制为 {max_samples} 个样本")
        data_dir = resolve_inside_project(config['data'].get('data_dir', 'data/finmme'))
        finmme_data = load_finmme_data(
            data_dir=str(data_dir),
            splits=['train', 'test'],
            feature_dim=config['data'].get('feature_dim', 768),
            use_pretrained_features=config['data'].get('use_pretrained_features', True),
            max_samples=max_samples
        )
        
        # Get modalities and labels
        train_mods = finmme_data['train']['modalities']
        full_train_labels = finmme_data['train']['labels']
        
        # Split train into train/val
        n_samples = len(full_train_labels)
        n_train = int(n_samples * config['data']['train_ratio'])
        n_val = n_samples - n_train
        
        # Ensure validation set is not empty
        if n_val == 0:
            print(f"Warning: Validation set is empty (n_samples={n_samples}, train_ratio={config['data']['train_ratio']})")
            print("Adjusting: using at least 1 sample for validation")
            n_val = 1
            n_train = n_samples - n_val
        
        full_train_mods = train_mods
        train_mods = [mod[:n_train] for mod in full_train_mods]
        val_mods = [mod[n_train:] for mod in full_train_mods]
        test_mods = finmme_data['test']['modalities']
        
        train_labels = full_train_labels[:n_train]
        val_labels = full_train_labels[n_train:]
        test_labels = finmme_data['test']['labels']
        
        # Update config with actual data dimensions
        config['data']['n_modalities'] = len(train_mods)
        config['data']['feature_dims'] = [mod.shape[-1] for mod in train_mods]
        config['data']['seq_lengths'] = [mod.shape[1] for mod in train_mods]
        config['data']['output_dim'] = 1  # Regression task
        
        print(f"FinMME data loaded:")
        print(f"  Train: {len(train_labels)} samples")
        print(f"  Val: {len(val_labels)} samples")
        print(f"  Test: {len(test_labels)} samples")
        print(f"  Feature dims: {config['data']['feature_dims']}")
        
    elif data_source == 'finmultitime':
        print("Loading FinMultiTime dataset...")
        market = config['data'].get('market', 'SP500')
        max_samples = config['data'].get('max_samples', None)
        if max_samples:
            print(f"快速测试模式: 限制为 {max_samples} 个样本")
        data_dir = resolve_inside_project(config['data'].get('data_dir', 'data/finmultitime'))
        finmultitime_data = load_finmultitime_data(
            data_dir=str(data_dir),
            splits=['train', 'test'],
            feature_dim=config['data'].get('feature_dim', 768),
            use_pretrained_features=config['data'].get('use_pretrained_features', True),
            market=market,
            max_samples=max_samples
        )
        
        # Get modalities and labels
        train_mods = finmultitime_data['train']['modalities']
        full_train_labels = finmultitime_data['train']['labels']
        
        # Split train into train/val
        n_samples = len(full_train_labels)
        n_train = int(n_samples * config['data']['train_ratio'])
        n_val = n_samples - n_train
        
        if n_val == 0:
            print(f"Warning: Validation set is empty")
            n_val = 1
            n_train = n_samples - n_val
        
        full_train_mods = train_mods
        train_mods = [mod[:n_train] for mod in full_train_mods]
        val_mods = [mod[n_train:] for mod in full_train_mods]
        test_mods = finmultitime_data['test']['modalities']
        
        train_labels = full_train_labels[:n_train]
        val_labels = full_train_labels[n_train:]
        test_labels = finmultitime_data['test']['labels']
        
        # Update config with actual data dimensions
        config['data']['n_modalities'] = len(train_mods)
        config['data']['feature_dims'] = [mod.shape[-1] for mod in train_mods]
        config['data']['seq_lengths'] = [mod.shape[1] for mod in train_mods]
        config['data']['output_dim'] = 1  # Regression task
        
        print(f"FinMultiTime data loaded ({market}):")
        print(f"  Train: {len(train_labels)} samples")
        print(f"  Val: {len(val_labels)} samples")
        print(f"  Test: {len(test_labels)} samples")
        print(f"  Feature dims: {config['data']['feature_dims']}")
        
    elif data_source == 'fcmr':
        print("Loading FCMR dataset...")
        # Default: original multi-label setup, but allow alternative tasks via config.task.name
        task_cfg = config.get('task', {}) or {}
        task_name = task_cfg.get('name', 'original')
        task_kwargs = task_cfg.get('kwargs', None)
        task_info = get_fcmr_task_info(task_name)
        task_type = task_info['task_type']
        is_multilabel = task_info.get('is_multilabel', False)
        difficulty = config['data'].get('difficulty', 'all')
        max_samples = config['data'].get('max_samples', None)
        if max_samples:
            print(f"快速测试模式: 限制为 {max_samples} 个样本")
        print(f"FCMR task: {task_name} ({task_info.get('description','')})")
        data_dir = resolve_inside_project(config['data'].get('data_dir', 'data/fcmr'))
        fcmr_data = load_fcmr_data(
            data_dir=str(data_dir),
            splits=['train', 'test'],
            difficulty=difficulty,
            feature_dim=config['data'].get('feature_dim', 768),
            use_pretrained_features=config['data'].get('use_pretrained_features', True),
            max_samples=max_samples,
            task=task_name,
            task_kwargs=task_kwargs
        )
        
        # Get modalities and labels
        train_mods = fcmr_data['train']['modalities']
        full_train_labels = fcmr_data['train']['labels']
        
        # Split train into train/val
        n_samples = len(full_train_labels)
        n_train = int(n_samples * config['data']['train_ratio'])
        n_val = n_samples - n_train
        
        if n_val == 0:
            print(f"Warning: Validation set is empty")
            n_val = 1
            n_train = n_samples - n_val
        
        full_train_mods = train_mods
        train_mods = [mod[:n_train] for mod in full_train_mods]
        val_mods = [mod[n_train:] for mod in full_train_mods]
        test_mods = fcmr_data['test']['modalities']
        
        train_labels = full_train_labels[:n_train]
        val_labels = full_train_labels[n_train:]
        test_labels = fcmr_data['test']['labels']
        
        # Update config with actual data dimensions
        config['data']['n_modalities'] = len(train_mods)
        config['data']['feature_dims'] = [mod.shape[-1] for mod in train_mods]
        config['data']['seq_lengths'] = [mod.shape[1] for mod in train_mods]
        # Ensure output dim matches task definition
        config['data']['output_dim'] = int(task_info.get('output_dim', 8))
        
        print(f"FCMR data loaded (difficulty: {difficulty}):")
        print(f"  Train: {len(train_labels)} samples")
        print(f"  Val: {len(val_labels)} samples")
        print(f"  Test: {len(test_labels)} samples")
        print(f"  Feature dims: {config['data']['feature_dims']}")
        print(f"  Output dim: {config['data']['output_dim']}")
    
    elif data_source == 'a_share':
        print("Loading A-share CSV dataset (price/macro/text)...")
        data_cfg = config['data']
        data_dir = resolve_inside_project(data_cfg.get('data_dir', 'data_a_share'))
        window = int(data_cfg.get('window', 30))
        lead = int(data_cfg.get('lead', 1))
        split_dates = data_cfg.get('split_dates', None)
        if split_dates is None:
            raise ValueError("For data.source == 'a_share', 'split_dates' must be provided in config['data'].")

        a_share_data = load_a_share_data(
            data_dir=str(data_dir),
            window=window,
            lead=lead,
            split_dates=split_dates,
        )

        # Extract modalities and labels for each split
        train_mods = a_share_data['train']['modalities']
        val_mods = a_share_data['val']['modalities']
        test_mods = a_share_data['test']['modalities']

        train_labels = a_share_data['train']['labels']
        val_labels = a_share_data['val']['labels']
        test_labels = a_share_data['test']['labels']

        # Update config with actual data dimensions
        config['data']['n_modalities'] = len(train_mods)
        config['data']['feature_dims'] = [mod.shape[-1] for mod in train_mods]
        config['data']['seq_lengths'] = [mod.shape[1] for mod in train_mods]
        config['data']['output_dim'] = 1  # Regression on future return

        print("A-share data loaded:")
        print(f"  Train: {len(train_labels)} samples")
        print(f"  Val: {len(val_labels)} samples")
        print(f"  Test: {len(test_labels)} samples")
        print(f"  Feature dims: {config['data']['feature_dims']}")
        print(f"  Seq lengths: {config['data']['seq_lengths']}")
        
    else:
        # Generate synthetic data
        print("Generating synthetic data...")
        modalities, labels = generate_synthetic_data(
            n_samples=config['data']['n_samples'],
            n_modalities=config['data']['n_modalities'],
            seq_lengths=config['data']['seq_lengths'],
            feature_dims=config['data']['feature_dims'],
            output_dim=config['data']['output_dim']
        )
        
        # Split data
        n_samples = len(labels)
        n_train = int(n_samples * config['data']['train_ratio'])
        n_val = int(n_samples * config['data']['val_ratio'])
        
        # Ensure validation set is not empty
        if n_val == 0:
            print(f"Warning: Validation set is empty (n_samples={n_samples}, val_ratio={config['data']['val_ratio']})")
            print("Adjusting: using at least 1 sample for validation")
            n_val = 1
            n_train = min(n_train, n_samples - n_val - 1)  # Ensure test set has at least 1 sample
        
        train_mods = [mod[:n_train] for mod in modalities]
        val_mods = [mod[n_train:n_train+n_val] for mod in modalities]
        test_mods = [mod[n_train+n_val:] for mod in modalities]
        
        train_labels = labels[:n_train]
        val_labels = labels[n_train:n_train+n_val]
        test_labels = labels[n_train+n_val:]
        
        # Print data diagnostics for synthetic data
        print(f"\n=== Synthetic Data Diagnostics ===")
        print(f"Train labels - mean: {np.mean(train_labels):.4f}, std: {np.std(train_labels):.4f}")
        print(f"Val labels - mean: {np.mean(val_labels):.4f}, std: {np.std(val_labels):.4f}")
        print(f"Test labels - mean: {np.mean(test_labels):.4f}, std: {np.std(test_labels):.4f}")
        for i, mod in enumerate(train_mods):
            print(f"Modality {i+1} - shape: {mod.shape}, mean: {mod.mean():.4f}, std: {mod.std():.4f}")
    
    # Create dataloaders
    label_dtype = "float"
    if task_type == "classification" and not is_multilabel:
        label_dtype = "long"

    train_loader = get_dataloader(
        train_mods, train_labels,
        batch_size=config['data']['batch_size'],
        shuffle=True,
        label_dtype=label_dtype
    )
    val_loader = get_dataloader(
        val_mods, val_labels,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        label_dtype=label_dtype
    )
    test_loader = get_dataloader(
        test_mods, test_labels,
        batch_size=config['data']['batch_size'],
        shuffle=False,
        label_dtype=label_dtype
    )
    
    # Warn if any dataloader is empty
    if len(train_loader) == 0:
        print("Warning: Training dataloader is empty!")
    if len(val_loader) == 0:
        print("Warning: Validation dataloader is empty! Training will continue but validation metrics will be default values.")
    if len(test_loader) == 0:
        print("Warning: Test dataloader is empty!")
    
    input_dims = config['data']['feature_dims']
    output_dim = config['data']['output_dim']
    
    # Model configurations
    model_configs = {
        'TFN': {
            'class': TFN,
            'args': {
                'input_dims': input_dims,
                'hidden_dim': config['model']['tfn']['hidden_dim'],
                'output_dim': output_dim,
                'dropout': config['model']['tfn']['dropout']
            }
        },
        'LMF': {
            'class': LMF,
            'args': {
                'input_dims': input_dims,
                'hidden_dim': config['model']['lmf']['hidden_dim'],
                'output_dim': output_dim,
                'rank': config['model']['lmf']['rank'],
                'dropout': config['model']['lmf']['dropout']
            }
        },
        'MFN': {
            'class': MFN,
            'args': {
                'input_dims': input_dims,
                'hidden_dim': config['model']['mfn']['hidden_dim'],
                'output_dim': output_dim,
                'memory_size': config['model']['mfn']['memory_size'],
                'num_layers': config['model']['mfn']['num_layers'],
                'dropout': config['model']['mfn']['dropout']
            }
        },
        'MulT': {
            'class': MulT,
            'args': {
                'input_dims': input_dims,
                'd_model': config['model']['mult']['d_model'],
                'output_dim': output_dim,
                'num_heads': config['model']['mult']['num_heads'],
                'num_layers': config['model']['mult']['num_layers'],
                'dropout': config['model']['mult']['dropout']
            }
        },
        'GCN': {
            'class': GCNFusion,
            'args': {
                'input_dims': input_dims,
                'hidden_dim': config['model']['gcn']['hidden_dim'],
                'output_dim': output_dim,
                'num_layers': config['model']['gcn']['num_layers'],
                'dropout': config['model']['gcn']['dropout']
            }
        },
        'Hypergraph': {
            'class': HypergraphFusion,
            'args': {
                'input_dims': input_dims,
                'hidden_dim': config['model']['hypergraph']['hidden_dim'],
                'output_dim': output_dim,
                'num_layers': config['model']['hypergraph']['num_layers'],
                'dropout': config['model']['hypergraph']['dropout']
            }
        },
        'QuantumHybrid': {
            'class': QuantumHybridModel,
            'args': {
                'input_dims': input_dims,
                'hidden_dim': config['model']['quantum']['hidden_dim'],
                'output_dim': output_dim,
                'n_qubits': config['model']['quantum']['n_qubits'],
                'n_quantum_layers': config['model']['quantum']['n_quantum_layers'],
                'dropout': config['model']['quantum']['dropout']
            }
        }
    }
    
    # Create save directory
    save_dir = config['training']['save_dir']
    os.makedirs(save_dir, exist_ok=True)
    
    # Train models
    results = {}
    models_to_train = config['models_to_train']
    
    for model_name in models_to_train:
        if model_name not in model_configs:
            print(f"Warning: {model_name} not found in model configs, skipping...")
            continue
        
        try:
            # Create model
            model_class = model_configs[model_name]['class']
            model_args = model_configs[model_name]['args']
            model = model_class(**model_args).to(device)
            
            # Train
            trained_model, tracker, val_metrics = train_model(
                model_name, model, train_loader, val_loader, config, device, task_type=task_type, is_multilabel=is_multilabel
            )
            
            # Test
            if task_type == 'classification':
                criterion = nn.BCEWithLogitsLoss() if is_multilabel else nn.CrossEntropyLoss()
            else:
                criterion = nn.MSELoss()
            test_loss, test_metrics = validate(trained_model, test_loader, criterion, device, task_type=task_type, is_multilabel=is_multilabel)
            
            # Save model
            model_path = os.path.join(save_dir, f'{model_name.lower()}_best.pt')
            torch.save(trained_model.state_dict(), model_path)
            
            # Save results (convert numpy types to Python native types)
            results[model_name] = {
                'val_metrics': val_metrics,
                'test_metrics': test_metrics,
                'best_val_loss': float(tracker.val_losses[tracker.get_best_epoch()]),
                'best_epoch': int(tracker.get_best_epoch() + 1)
            }
            
            # Save plots
            tracker.plot_losses(os.path.join(save_dir, f'{model_name.lower()}_losses.png'))
            metric_name = 'R2' if task_type == 'regression' else ('F1_Micro' if 'F1_Micro' in val_metrics else None)
            if metric_name:
                tracker.plot_metrics(metric_name, os.path.join(save_dir, f'{model_name.lower()}_{metric_name.lower()}.png'))
            
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save all results
    results_path = os.path.join(save_dir, 'all_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print("Training completed!")
    print(f"Results saved to {save_dir}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()

