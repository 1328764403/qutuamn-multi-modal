"""
加载真实数据集的示例
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path


def load_from_csv(csv_paths, label_path=None):
    """
    从CSV文件加载多模态数据
    
    Args:
        csv_paths: list of paths to CSV files, each CSV contains one modality
        label_path: path to labels CSV file
    Returns:
        modalities: list of numpy arrays
        labels: numpy array
    """
    modalities = []
    
    # Load each modality
    for csv_path in csv_paths:
        df = pd.read_csv(csv_path)
        # Assume first column is index/time, rest are features
        features = df.iloc[:, 1:].values  # (n_samples, feature_dim)
        modalities.append(features)
    
    # Load labels
    if label_path:
        labels_df = pd.read_csv(label_path)
        labels = labels_df.iloc[:, -1].values  # Last column as labels
    else:
        # Generate dummy labels if not provided
        labels = np.random.randn(len(modalities[0]), 1)
    
    return modalities, labels


def load_from_numpy(npz_path):
    """
    从NPZ文件加载数据
    
    Args:
        npz_path: path to .npz file containing 'modalities' and 'labels' keys
    Returns:
        modalities: list of numpy arrays
        labels: numpy array
    """
    data = np.load(npz_path, allow_pickle=True)
    modalities = [data[f'modality_{i}'] for i in range(len([k for k in data.keys() if k.startswith('modality')]))]
    labels = data['labels']
    return modalities, labels


def load_from_h5(h5_path):
    """
    从HDF5文件加载数据
    
    Args:
        h5_path: path to .h5 file
    Returns:
        modalities: list of numpy arrays
        labels: numpy array
    """
    import h5py
    
    with h5py.File(h5_path, 'r') as f:
        modalities = []
        i = 0
        while f'modality_{i}' in f.keys():
            modalities.append(f[f'modality_{i}'][:])
            i += 1
        labels = f['labels'][:]
    
    return modalities, labels


def load_time_series_data(data_dir, modalities=['modality1', 'modality2', 'modality3']):
    """
    加载时间序列多模态数据
    
    Args:
        data_dir: directory containing data files
        modalities: list of modality names (file names)
    Returns:
        modalities: list of numpy arrays, each (n_samples, seq_len, feature_dim)
        labels: numpy array, shape (n_samples, output_dim)
    """
    data_dir = Path(data_dir)
    modalities_data = []
    
    for mod_name in modalities:
        # Load modality data (assuming CSV format)
        mod_path = data_dir / f'{mod_name}.csv'
        if mod_path.exists():
            df = pd.read_csv(mod_path)
            # Convert to numpy array
            mod_data = df.values  # (n_samples, seq_len, feature_dim) or (n_samples, feature_dim)
            modalities_data.append(mod_data)
        else:
            raise FileNotFoundError(f"Modality file not found: {mod_path}")
    
    # Load labels
    label_path = data_dir / 'labels.csv'
    if label_path.exists():
        labels_df = pd.read_csv(label_path)
        labels = labels_df.values
    else:
        # Generate dummy labels
        labels = np.random.randn(len(modalities_data[0]), 1)
    
    return modalities_data, labels


# 示例：加载CMU-MOSI数据集格式的数据
def load_cmu_mosi_style(data_dir):
    """
    加载CMU-MOSI风格的多模态数据
    假设数据格式：
    - text_features.csv: 文本特征
    - audio_features.csv: 音频特征  
    - video_features.csv: 视频特征
    - labels.csv: 标签
    """
    data_dir = Path(data_dir)
    
    modalities = []
    for mod_name in ['text_features', 'audio_features', 'video_features']:
        mod_path = data_dir / f'{mod_name}.csv'
        if mod_path.exists():
            df = pd.read_csv(mod_path)
            modalities.append(df.values)
    
    label_path = data_dir / 'labels.csv'
    if label_path.exists():
        labels = pd.read_csv(label_path).values
    else:
        labels = np.random.randn(len(modalities[0]), 1)
    
    return modalities, labels







