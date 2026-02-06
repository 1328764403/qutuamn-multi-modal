"""
Metrics for model evaluation
"""

import torch
import numpy as np
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, f1_score, precision_score, recall_score,
    hamming_loss, classification_report
)
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_metrics(y_true, y_pred, task_type='regression', verbose=False, is_multilabel=False):
    """
    Calculate metrics for regression or classification tasks
    
    Args:
        y_true: true values (numpy array)
        y_pred: predicted values (numpy array)
        task_type: 'regression' or 'classification'
        is_multilabel: whether classification is multilabel
        verbose: if True, print diagnostic information
    Returns:
        dict of metrics
    """
    if task_type == 'classification':
        return calculate_classification_metrics(y_true, y_pred, is_multilabel=is_multilabel)
    else:
        return calculate_regression_metrics(y_true, y_pred, verbose=verbose)


def calculate_regression_metrics(y_true, y_pred, verbose=False):
    """
    Calculate regression metrics with improved R² calculation and diagnostics
    
    Args:
        y_true: true values (numpy array)
        y_pred: predicted values (numpy array)
        verbose: if True, print diagnostic information
    Returns:
        dict of metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Data validation
    if len(y_true) == 0 or len(y_pred) == 0:
        raise ValueError("Empty arrays provided for metrics calculation")
    
    if len(y_true) != len(y_pred):
        raise ValueError(f"Length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}")
    
    # Check for constant values (which can cause R² issues)
    y_true_std = np.std(y_true)
    y_pred_std = np.std(y_pred)
    
    if verbose:
        print(f"  Data diagnostics:")
        print(f"    y_true: mean={np.mean(y_true):.4f}, std={y_true_std:.4f}, "
              f"min={np.min(y_true):.4f}, max={np.max(y_true):.4f}")
        print(f"    y_pred: mean={np.mean(y_pred):.4f}, std={y_pred_std:.4f}, "
              f"min={np.min(y_pred):.4f}, max={np.max(y_pred):.4f}")
    
    # Handle constant y_true (R² is undefined, return 0)
    if y_true_std < 1e-8:
        if verbose:
            print(f"  Warning: y_true is constant (std={y_true_std:.8f}), R² set to 0")
        r2 = 0.0
    else:
        # Use sklearn's r2_score with force_finite=True (default)
        r2 = r2_score(y_true, y_pred)
        
        if verbose and r2 < 0:
            # Calculate components for debugging
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            print(f"  R² diagnostic (negative R²={r2:.4f}):")
            print(f"    SS_res (residual sum of squares) = {ss_res:.4f}")
            print(f"    SS_tot (total sum of squares) = {ss_tot:.4f}")
            print(f"    SS_res/SS_tot = {ss_res/ss_tot:.4f} > 1.0 (model worse than mean)")
            print(f"    Mean baseline prediction error: {np.sqrt(ss_tot/len(y_true)):.4f}")
            print(f"    Model prediction error (RMSE): {np.sqrt(ss_res/len(y_true)):.4f}")
    
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    
    # Mean absolute percentage error
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-8))) * 100
    
    # Convert numpy types to Python native types for JSON serialization
    return {
        'MSE': float(mse),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'R2': float(r2),
        'MAPE': float(mape)
    }


def calculate_classification_metrics(y_true, y_pred, is_multilabel=False):
    """
    Calculate classification metrics
    
    Args:
        y_true: true labels (numpy array), shape (n_samples, n_classes) for multilabel
        y_pred: predicted labels (numpy array), shape (n_samples, n_classes) for multilabel
        is_multilabel: whether this is a multilabel classification task
    Returns:
        dict of metrics
    """
    if is_multilabel:
        # Multi-label classification
        # Convert probabilities to binary predictions
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            # If predictions are probabilities, convert to binary
            y_pred_binary = (y_pred > 0.5).astype(int)
        else:
            y_pred_binary = y_pred
        
        # Ensure same shape
        if y_true.shape != y_pred_binary.shape:
            # If y_pred is one-hot, convert to binary
            if y_pred_binary.ndim == 1:
                y_pred_binary = (y_pred_binary > 0.5).astype(int)
                # Expand to match y_true
                if y_true.ndim == 2:
                    y_pred_binary = np.eye(y_true.shape[1])[y_pred_binary]
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred_binary)
        f1_macro = f1_score(y_true, y_pred_binary, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred_binary, average='micro', zero_division=0)
        precision = precision_score(y_true, y_pred_binary, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred_binary, average='macro', zero_division=0)
        hamming = hamming_loss(y_true, y_pred_binary)
        
        return {
            'Accuracy': float(accuracy),
            'F1_Macro': float(f1_macro),
            'F1_Micro': float(f1_micro),
            'Precision': float(precision),
            'Recall': float(recall),
            'Hamming_Loss': float(hamming)
        }
    else:
        # Single-label classification
        # Convert to class indices if needed
        if y_pred.ndim == 2:
            y_pred = np.argmax(y_pred, axis=1)
        if y_true.ndim == 2:
            y_true = np.argmax(y_true, axis=1)
        
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        precision = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall = recall_score(y_true, y_pred, average='macro', zero_division=0)
        
        return {
            'Accuracy': float(accuracy),
            'F1_Macro': float(f1_macro),
            'F1_Micro': float(f1_micro),
            'Precision': float(precision),
            'Recall': float(recall)
        }


class MetricsTracker:
    """Track metrics during training"""
    
    def __init__(self):
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def update(self, train_loss, val_loss, train_metrics=None, val_metrics=None):
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        if train_metrics:
            self.train_metrics.append(train_metrics)
        if val_metrics:
            self.val_metrics.append(val_metrics)
    
    def get_best_epoch(self):
        """Get epoch with best validation loss"""
        return np.argmin(self.val_losses)
    
    def plot_losses(self, save_path=None):
        """Plot training and validation losses"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.train_losses, label='Train Loss')
        plt.plot(self.val_losses, label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()
    
    def plot_metrics(self, metric_name='R2', save_path=None):
        """Plot a specific metric"""
        if not self.train_metrics or not self.val_metrics:
            return
        
        train_vals = [m[metric_name] for m in self.train_metrics]
        val_vals = [m[metric_name] for m in self.val_metrics]
        
        plt.figure(figsize=(10, 6))
        plt.plot(train_vals, label=f'Train {metric_name}')
        plt.plot(val_vals, label=f'Val {metric_name}')
        plt.xlabel('Epoch')
        plt.ylabel(metric_name)
        plt.title(f'Training and Validation {metric_name}')
        plt.legend()
        plt.grid(True)
        if save_path:
            plt.savefig(save_path)
        plt.close()

