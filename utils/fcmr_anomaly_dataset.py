"""
FCMR 异常模态检测数据集
在正常FCMR数据基础上，人为制造各种模态异常
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, List, Dict
import random


class FCMRAnomalyDataset(Dataset):
    """
    FCMR异常模态检测数据集
    
    任务：检测多模态样本中是否存在异常模态
    标签格式（二分类）：
        0 = 正常样本（所有模态都正常）
        1 = 异常样本（至少一个模态异常）
    """
    
    def __init__(
        self,
        normal_features: List[np.ndarray],
        anomaly_type: str = 'binary',
        anomaly_ratio: float = 0.3,
        noise_std: float = 0.5,
        dropout_ratio: float = 0.5,
        random_seed: int = 42
    ):
        """
        Args:
            normal_features: 正常多模态特征列表 [n_samples, 3, 768]
            anomaly_type: 异常检测类型 ('binary' 或 'multilabel')
            anomaly_ratio: 异常样本比例
            noise_std: 噪声异常的标准差
            dropout_ratio: 模态缺失时保留特征的比例
            random_seed: 随机种子
        """
        self.normal_features = normal_features
        self.n_samples = len(normal_features)
        self.anomaly_type = anomaly_type
        self.anomaly_ratio = anomaly_ratio
        self.noise_std = noise_std
        self.dropout_ratio = dropout_ratio
        
        random.seed(random_seed)
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        
        # 生成异常样本的索引
        self.anomaly_indices = random.sample(
            range(self.n_samples), 
            k=int(self.n_samples * anomaly_ratio)
        )
        self.normal_indices = [i for i in range(self.n_samples) if i not in self.anomaly_indices]
        
        # 创建异常样本
        self.anomaly_samples = self._create_anomaly_samples()
    
    def _create_anomaly_samples(self) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
        """
        为每个异常样本生成异常模态
        
        异常类型：
        1. 噪声注入 - 用随机噪声替换模态特征
        2. 模态缺失 - 将特征置零或部分置零
        3. 特征置换 - 随机置换不同样本的模态特征
        """
        anomaly_samples = {}
        
        for idx in self.anomaly_indices:
            original = self.normal_features[idx].copy()  # [3, 768]
            
            # 随机选择异常类型和异常模态
            anomaly_modality = random.randint(0, 2)  # 0=text, 1=table, 2=chart
            method = random.choice(['noise', 'dropout', 'swap'])
            
            if method == 'noise':
                # 噪声注入
                noise = np.random.randn(768).astype(np.float32) * self.noise_std
                original[anomaly_modality] = noise
            
            elif method == 'dropout':
                # 模态缺失
                mask = np.random.random(768) > self.dropout_ratio
                original[anomaly_modality] = original[anomaly_modality] * mask.astype(np.float32)
            
            elif method == 'swap':
                # 特征置换
                swap_idx = random.choice([i for i in range(self.n_samples) if i != idx])
                original[anomaly_modality] = self.normal_features[swap_idx][anomaly_modality]
            
            # 生成标签
            if self.anomaly_type == 'binary':
                label = np.array([1], dtype=np.float32)
            else:
                label = np.zeros(3, dtype=np.float32)
                label[anomaly_modality] = 1.0
            
            anomaly_samples[idx] = (original, label)
        
        return anomaly_samples
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Tuple[List[torch.Tensor], torch.Tensor]:
        if idx in self.anomaly_samples:
            features, label = self.anomaly_samples[idx]
        else:
            features = self.normal_features[idx]
            if self.anomaly_type == 'binary':
                label = np.array([0], dtype=np.float32)
            else:
                label = np.zeros(3, dtype=np.float32)
        
        features = [torch.from_numpy(f).float() for f in features]
        label = torch.from_numpy(label).float()
        
        return features, label


def create_anomaly_dataloader(
    normal_features: List[np.ndarray],
    batch_size: int = 16,
    anomaly_type: str = 'binary',
    anomaly_ratio: float = 0.3,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    shuffle: bool = True,
    random_seed: int = 42
) -> Tuple:
    """创建异常检测的DataLoader"""
    full_dataset = FCMRAnomalyDataset(
        normal_features=normal_features,
        anomaly_type=anomaly_type,
        anomaly_ratio=anomaly_ratio,
        random_seed=random_seed
    )
    
    n_samples = len(full_dataset)
    n_train = int(n_samples * train_ratio)
    n_val = int(n_samples * val_ratio)
    n_test = n_samples - n_train - n_val
    
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
        full_dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(random_seed)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    if anomaly_type == 'binary':
        task_info = {
            'output_dim': 1,
            'task_type': 'classification',
            'is_multilabel': False,
            'description': '模态异常检测（二分类：正常/异常）'
        }
    else:
        task_info = {
            'output_dim': 3,
            'task_type': 'classification',
            'is_multilabel': True,
            'description': '模态异常检测（多标签：每个模态独立）'
        }
    
    return train_loader, val_loader, test_loader, task_info
