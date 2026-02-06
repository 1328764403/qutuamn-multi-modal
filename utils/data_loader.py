"""
Data loader for multimodal datasets
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class MultimodalDataset(Dataset):
    """
    Multimodal dataset
    Supports multiple modalities with different sequence lengths
    """
    
    def __init__(self, modalities, labels, seq_lengths=None, label_dtype: str = "float"):
        """
        Args:
            modalities: list of numpy arrays, each of shape (n_samples, seq_len, feature_dim)
            labels: numpy array of shape (n_samples, output_dim)
            seq_lengths: optional list of sequence lengths for each modality
        """
        self.modalities = modalities
        self.labels = labels
        self.seq_lengths = seq_lengths or [mod.shape[1] for mod in modalities]
        self.n_samples = len(labels)
        self.label_dtype = label_dtype
        
        assert all(mod.shape[0] == self.n_samples for mod in modalities), \
            "All modalities must have the same number of samples"
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, idx):
        mods = [torch.FloatTensor(mod[idx]) for mod in self.modalities]
        if self.label_dtype == "long":
            label = torch.tensor(self.labels[idx], dtype=torch.long).squeeze()
        else:
            label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return tuple(mods), label


def get_dataloader(modalities, labels, batch_size=32, shuffle=True, seq_lengths=None, label_dtype: str = "float"):
    """
    Create a DataLoader for multimodal data
    
    Args:
        modalities: list of numpy arrays
        labels: numpy array of labels
        batch_size: batch size
        shuffle: whether to shuffle
        seq_lengths: optional sequence lengths
    Returns:
        DataLoader
    """
    dataset = MultimodalDataset(modalities, labels, seq_lengths, label_dtype=label_dtype)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def generate_synthetic_data(n_samples=1000, n_modalities=3, seq_lengths=None, feature_dims=None, output_dim=1):
    """
    Generate synthetic multimodal data for testing
    
    Args:
        n_samples: number of samples
        n_modalities: number of modalities
        seq_lengths: list of sequence lengths for each modality
        feature_dims: list of feature dimensions for each modality
        output_dim: output dimension
    Returns:
        modalities: list of numpy arrays
        labels: numpy array of labels
    """
    if seq_lengths is None:
        seq_lengths = [10] * n_modalities
    if feature_dims is None:
        feature_dims = [32] * n_modalities
    
    modalities = []
    for seq_len, feat_dim in zip(seq_lengths, feature_dims):
        mod = np.random.randn(n_samples, seq_len, feat_dim)
        modalities.append(mod)
    
    # Generate labels using a simple, learnable linear combination
    # Reference: Standard synthetic regression data: y = Xw + b + ε
    # For multimodal: y = sum(weight_i * feature_i) + noise
    
    # Step 1: Extract simple features from each modality
    # Use mean pooling across sequence and feature dimensions to get a single feature per sample
    modality_features = []
    for mod in modalities:
        # Average pooling: (n_samples, seq_len, feat_dim) -> (n_samples,)
        mod_feature = mod.mean(axis=(1, 2))  # Simple mean across all dimensions
        modality_features.append(mod_feature)
    
    # Step 2: Create a simple linear combination with fixed weights
    # This ensures the relationship is learnable by neural networks
    labels = np.zeros((n_samples, output_dim))
    
    # Use fixed, positive weights that sum to a reasonable value
    weights = np.array([1.0, 0.8, 0.6][:n_modalities])  # Decreasing weights for each modality
    weights = weights / weights.sum() * 2.0  # Normalize and scale
    
    for i, mod_feature in enumerate(modality_features):
        weight = weights[i] if i < len(weights) else 0.5
        labels[:, 0] += weight * mod_feature
    
    # Step 3: Add small Gaussian noise (10% of signal std)
    signal_std = np.std(labels)
    noise_std = signal_std * 0.1  # 10% noise level
    labels += np.random.randn(n_samples, output_dim) * noise_std
    
    # Step 4: Ensure labels have reasonable scale (not too large/small)
    # This helps with training stability
    label_mean = np.mean(labels)
    label_std = np.std(labels)
    if label_std > 0:
        # Normalize to have mean ~0 and std ~1, then scale to reasonable range
        labels = (labels - label_mean) / (label_std + 1e-8)
        labels = labels * 2.0  # Scale to range roughly [-4, 4] for most samples
    
    return modalities, labels







