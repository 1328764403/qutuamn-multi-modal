"""
Extended metrics for various tasks beyond standard classification/regression
Includes ranking, retrieval, anomaly detection, and financial-specific metrics
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    ndcg_score, mean_reciprocal_rank,
    cohen_kappa_score, matthews_corrcoef,
    silhouette_score, calinski_harabasz_score,
    davies_bouldin_score
)
from scipy.stats import spearmanr, kendalltau, pearsonr
import torch
import torch.nn.functional as F


def calculate_ranking_metrics(y_true, y_pred, k=10):
    """
    Calculate ranking metrics (NDCG, MRR, MAP)
    
    Args:
        y_true: true relevance scores (n_samples, n_items) or (n_samples,)
        y_pred: predicted scores (n_samples, n_items) or (n_samples,)
        k: top-k for NDCG
    Returns:
        dict of ranking metrics
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
        y_true = y_true.reshape(-1, 1)
    
    n_samples = y_true.shape[0]
    
    # NDCG@k
    ndcg_scores = []
    for i in range(n_samples):
        try:
            ndcg = ndcg_score([y_true[i]], [y_pred[i]], k=min(k, len(y_true[i])))
            ndcg_scores.append(ndcg)
        except:
            ndcg_scores.append(0.0)
    
    # MRR (Mean Reciprocal Rank)
    mrr_scores = []
    for i in range(n_samples):
        true_rank = np.argsort(-y_true[i])
        pred_rank = np.argsort(-y_pred[i])
        # Find position of first relevant item
        for rank, idx in enumerate(pred_rank, 1):
            if y_true[i][idx] > 0:
                mrr_scores.append(1.0 / rank)
                break
        else:
            mrr_scores.append(0.0)
    
    # MAP (Mean Average Precision)
    map_scores = []
    for i in range(n_samples):
        relevant_items = np.where(y_true[i] > 0)[0]
        if len(relevant_items) == 0:
            map_scores.append(0.0)
            continue
        
        pred_rank = np.argsort(-y_pred[i])
        precisions = []
        relevant_found = 0
        for rank, idx in enumerate(pred_rank, 1):
            if idx in relevant_items:
                relevant_found += 1
                precisions.append(relevant_found / rank)
        map_scores.append(np.mean(precisions) if precisions else 0.0)
    
    return {
        'NDCG@{}'.format(k): float(np.mean(ndcg_scores)),
        'MRR': float(np.mean(mrr_scores)),
        'MAP': float(np.mean(map_scores))
    }


def calculate_correlation_metrics(y_true, y_pred):
    """
    Calculate correlation metrics (Pearson, Spearman, Kendall)
    
    Args:
        y_true: true values (n_samples,)
        y_pred: predicted values (n_samples,)
    Returns:
        dict of correlation metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Pearson correlation
    pearson_r, pearson_p = pearsonr(y_true, y_pred)
    
    # Spearman correlation
    spearman_r, spearman_p = spearmanr(y_true, y_pred)
    
    # Kendall's tau
    kendall_tau, kendall_p = kendalltau(y_true, y_pred)
    
    return {
        'Pearson_R': float(pearson_r),
        'Pearson_P': float(pearson_p),
        'Spearman_R': float(spearman_r),
        'Spearman_P': float(spearman_p),
        'Kendall_Tau': float(kendall_tau),
        'Kendall_P': float(kendall_p)
    }


def calculate_auc_metrics(y_true, y_pred, is_multilabel=False):
    """
    Calculate AUC-ROC and AUC-PR metrics
    
    Args:
        y_true: true labels (n_samples, n_classes) for multilabel or (n_samples,) for binary
        y_pred: predicted probabilities (n_samples, n_classes) or (n_samples,)
        is_multilabel: whether this is multilabel classification
    Returns:
        dict of AUC metrics
    """
    if is_multilabel:
        # Multi-label: calculate per-class AUC
        n_classes = y_true.shape[1]
        roc_aucs = []
        pr_aucs = []
        
        for i in range(n_classes):
            try:
                if len(np.unique(y_true[:, i])) > 1:  # Skip if all same class
                    roc_auc = roc_auc_score(y_true[:, i], y_pred[:, i])
                    pr_auc = average_precision_score(y_true[:, i], y_pred[:, i])
                    roc_aucs.append(roc_auc)
                    pr_aucs.append(pr_auc)
            except:
                pass
        
        return {
            'AUC-ROC_Macro': float(np.mean(roc_aucs)) if roc_aucs else 0.0,
            'AUC-PR_Macro': float(np.mean(pr_aucs)) if pr_aucs else 0.0,
            'AUC-ROC_Micro': roc_auc_score(y_true.flatten(), y_pred.flatten()) if len(np.unique(y_true.flatten())) > 1 else 0.0,
            'AUC-PR_Micro': average_precision_score(y_true.flatten(), y_pred.flatten()) if len(np.unique(y_true.flatten())) > 1 else 0.0
        }
    else:
        # Binary or multi-class
        if y_pred.ndim == 2:
            # Multi-class: use one-vs-rest
            n_classes = y_pred.shape[1]
            if n_classes == 2:
                # Binary classification
                roc_auc = roc_auc_score(y_true, y_pred[:, 1])
                pr_auc = average_precision_score(y_true, y_pred[:, 1])
            else:
                # Multi-class: macro average
                roc_aucs = []
                pr_aucs = []
                for i in range(n_classes):
                    try:
                        y_true_binary = (y_true == i).astype(int)
                        if len(np.unique(y_true_binary)) > 1:
                            roc_aucs.append(roc_auc_score(y_true_binary, y_pred[:, i]))
                            pr_aucs.append(average_precision_score(y_true_binary, y_pred[:, i]))
                    except:
                        pass
                roc_auc = np.mean(roc_aucs) if roc_aucs else 0.0
                pr_auc = np.mean(pr_aucs) if pr_aucs else 0.0
        else:
            # Binary: single probability
            roc_auc = roc_auc_score(y_true, y_pred)
            pr_auc = average_precision_score(y_true, y_pred)
        
        return {
            'AUC-ROC': float(roc_auc),
            'AUC-PR': float(pr_auc)
        }


def calculate_topk_accuracy(y_true, y_pred, k=5):
    """
    Calculate Top-K accuracy
    
    Args:
        y_true: true labels (n_samples,)
        y_pred: predicted probabilities (n_samples, n_classes)
        k: top-k
    Returns:
        float: top-k accuracy
    """
    if y_pred.ndim == 1:
        y_pred = y_pred.reshape(-1, 1)
    
    topk_pred = np.argsort(y_pred, axis=1)[:, -k:]
    correct = np.sum([y_true[i] in topk_pred[i] for i in range(len(y_true))])
    return correct / len(y_true)


def calculate_agreement_metrics(y_true, y_pred):
    """
    Calculate agreement metrics (Cohen's Kappa, Matthews Correlation Coefficient)
    
    Args:
        y_true: true labels (n_samples,)
        y_pred: predicted labels (n_samples,)
    Returns:
        dict of agreement metrics
    """
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    return {
        'Cohen_Kappa': float(kappa),
        'MCC': float(mcc)
    }


def calculate_clustering_metrics(X, labels):
    """
    Calculate clustering quality metrics
    
    Args:
        X: feature matrix (n_samples, n_features)
        labels: cluster labels (n_samples,)
    Returns:
        dict of clustering metrics
    """
    if len(np.unique(labels)) < 2:
        return {
            'Silhouette': 0.0,
            'Calinski_Harabasz': 0.0,
            'Davies_Bouldin': float('inf')
        }
    
    silhouette = silhouette_score(X, labels)
    calinski = calinski_harabasz_score(X, labels)
    davies_bouldin = davies_bouldin_score(X, labels)
    
    return {
        'Silhouette': float(silhouette),
        'Calinski_Harabasz': float(calinski),
        'Davies_Bouldin': float(davies_bouldin)
    }


def calculate_financial_metrics(y_true, y_pred, returns=None):
    """
    Calculate financial-specific metrics
    
    Args:
        y_true: true values (n_samples,)
        y_pred: predicted values (n_samples,)
        returns: actual returns (optional, n_samples,)
    Returns:
        dict of financial metrics
    """
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Sharpe Ratio (if returns provided)
    sharpe = None
    if returns is not None:
        returns = returns.flatten()
        excess_returns = returns - np.mean(returns)
        sharpe = np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)
    
    # Information Ratio
    prediction_error = y_pred - y_true
    ir = np.mean(prediction_error) / (np.std(prediction_error) + 1e-8)
    
    # Hit Rate (for directional prediction)
    if len(y_true) > 1:
        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0
        hit_rate = np.mean(true_direction == pred_direction)
    else:
        hit_rate = 0.0
    
    # Maximum Drawdown (if returns provided)
    mdd = None
    if returns is not None:
        cumulative = np.cumprod(1 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (cumulative - running_max) / running_max
        mdd = np.min(drawdown)
    
    metrics = {
        'Information_Ratio': float(ir),
        'Hit_Rate': float(hit_rate)
    }
    
    if sharpe is not None:
        metrics['Sharpe_Ratio'] = float(sharpe)
    if mdd is not None:
        metrics['Max_Drawdown'] = float(mdd)
    
    return metrics


def calculate_modality_importance_metrics(modality_features, labels, n_modalities=3):
    """
    Calculate metrics for modality importance analysis
    
    Args:
        modality_features: list of modality feature arrays, each (n_samples, feature_dim)
        labels: labels (n_samples,)
        n_modalities: number of modalities
    Returns:
        dict of modality importance metrics
    """
    importance_scores = []
    
    for i, mod_feat in enumerate(modality_features):
        # Calculate correlation between modality features and labels
        mod_flat = mod_feat.reshape(mod_feat.shape[0], -1)
        correlations = []
        for j in range(min(10, mod_flat.shape[1])):  # Sample features
            try:
                corr, _ = pearsonr(mod_flat[:, j], labels.flatten())
                correlations.append(abs(corr))
            except:
                pass
        importance_scores.append(np.mean(correlations) if correlations else 0.0)
    
    # Normalize to sum to 1
    total = sum(importance_scores)
    if total > 0:
        importance_scores = [s / total for s in importance_scores]
    
    return {
        f'Modality_{i+1}_Importance': float(importance_scores[i]) 
        for i in range(n_modalities)
    }


def calculate_all_extended_metrics(y_true, y_pred, task_type='classification', 
                                   is_multilabel=False, modality_features=None,
                                   returns=None, k=5):
    """
    Calculate all extended metrics for a given task
    
    Args:
        y_true: true values/labels
        y_pred: predicted values/probabilities
        task_type: 'classification', 'regression', 'ranking', 'clustering'
        is_multilabel: whether classification is multilabel
        modality_features: list of modality features (for importance analysis)
        returns: returns for financial metrics (optional)
        k: top-k for ranking/top-k accuracy
    Returns:
        dict of all metrics
    """
    all_metrics = {}
    
    if task_type == 'classification':
        # AUC metrics
        auc_metrics = calculate_auc_metrics(y_true, y_pred, is_multilabel)
        all_metrics.update(auc_metrics)
        
        # Top-K accuracy
        if y_pred.ndim == 2:
            y_true_class = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
            topk_acc = calculate_topk_accuracy(y_true_class, y_pred, k=k)
            all_metrics[f'Top-{k}_Accuracy'] = float(topk_acc)
        
        # Agreement metrics
        if not is_multilabel:
            y_true_class = np.argmax(y_true, axis=1) if y_true.ndim == 2 else y_true
            y_pred_class = np.argmax(y_pred, axis=1) if y_pred.ndim == 2 else y_pred
            agreement = calculate_agreement_metrics(y_true_class, y_pred_class)
            all_metrics.update(agreement)
    
    elif task_type == 'regression':
        # Correlation metrics
        corr_metrics = calculate_correlation_metrics(y_true, y_pred)
        all_metrics.update(corr_metrics)
        
        # Financial metrics (if returns provided)
        if returns is not None:
            financial_metrics = calculate_financial_metrics(y_true, y_pred, returns)
            all_metrics.update(financial_metrics)
    
    elif task_type == 'ranking':
        # Ranking metrics
        ranking_metrics = calculate_ranking_metrics(y_true, y_pred, k=k)
        all_metrics.update(ranking_metrics)
    
    # Modality importance (if features provided)
    if modality_features is not None:
        importance_metrics = calculate_modality_importance_metrics(
            modality_features, y_true, n_modalities=len(modality_features)
        )
        all_metrics.update(importance_metrics)
    
    return all_metrics
