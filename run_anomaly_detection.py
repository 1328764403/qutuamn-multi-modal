"""
FCMR 异常模态检测
用法: python run实验脚本_anomaly_detection.py --config configs/config_fcmr_anomaly.yaml
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import random
import argparse
import json
from pathlib import Path

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TFN, LMF, MFN, MulT, GCNFusion, HypergraphFusion, QuantumHybridModel
from utils.fcmr_anomaly_dataset import FCMRAnomalyDataset
from utils.metrics import calculate_metrics


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_epoch(model, dataloader, criterion, optimizer, device, is_multilabel=False):
    model.train()
    total_loss = 0
    all_preds, all_labels = [], []

    for features, labels in dataloader:
        features = [f.to(device) for f in features]
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(*features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.float32)
    all_preds = 1.0 / (1.0 + np.exp(-np.clip(all_preds, -50, 50)))

    metrics = calculate_metrics(all_labels, all_preds, task_type='classification', is_multilabel=is_multilabel)
    return total_loss / len(dataloader), metrics


def validate(model, dataloader, criterion, device, is_multilabel=False):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for features, labels in dataloader:
            features = [f.to(device) for f in features]
            labels = labels.to(device)

            outputs = model(*features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            all_preds.append(outputs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())

    all_preds = np.concatenate(all_preds).astype(np.float32)
    all_labels = np.concatenate(all_labels).astype(np.float32)
    all_preds = 1.0 / (1.0 + np.exp(-np.clip(all_preds, -50, 50)))

    metrics = calculate_metrics(all_labels, all_preds, task_type='classification', is_multilabel=is_multilabel)
    return total_loss / len(dataloader), metrics


def main():
    parser = argparse.ArgumentParser(description='FCMR 异常模态检测实验')
    parser.add_argument('--config', type=str, default='configs/config_fcmr_anomaly.yaml')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--save_dir', type=str, default='results/fcmr_anomaly')
    parser.add_argument('--anomaly_ratio', type=float, default=0.3)
    parser.add_argument('--anomaly_type', type=str, default='binary', choices=['binary', 'multilabel'])
    parser.add_argument('--data_dir', type=str, default='data/fcmr')
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")

    # 尝试加载FCMR数据，如果失败则使用合成数据
    print("加载数据...")
    try:
        from utils.load_fcmr import load_fcmr_data
        train_data, val_data, test_data = load_fcmr_data(args.data_dir, difficulty='all')
        
        def extract_features(data_list):
            features = []
            for item in data_list:
                if isinstance(item, dict) and 'features' in item:
                    feats = item['features']
                    if isinstance(feats, list) and len(feats) >= 3:
                        features.append(np.array(feats[:3], dtype=np.float32))
            return features
        
        all_features = extract_features(train_data) + extract_features(val_data) + extract_features(test_data)
        
        if len(all_features) == 0:
            raise ValueError("无法提取特征")
        print(f"加载了 {len(all_features)} 个样本")
    except Exception as e:
        print(f"加载FCMR数据失败: {e}，使用合成数据")
        all_features = [np.random.randn(3, 768).astype(np.float32) for _ in range(200)]

    is_multilabel = (args.anomaly_type == 'multilabel')
    output_dim = 3 if is_multilabel else 1

    # 创建异常检测数据集
    print(f"创建{args.anomaly_type}异常检测数据集...")
    full_dataset = FCMRAnomalyDataset(
        normal_features=all_features,
        anomaly_type=args.anomaly_type,
        anomaly_ratio=args.anomaly_ratio,
        random_seed=args.seed
    )

    n_total = len(full_dataset)
    n_train, n_val = int(n_total * 0.8), int(n_total * 0.1)
    n_test = n_total - n_train - n_val

    train_ds, val_ds, test_ds = random_split(full_dataset, [n_train, n_val, n_test],
                                               generator=torch.Generator().manual_seed(args.seed))
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size)

    # 模型
    input_dims = [768, 768, 768]
    models = {
        'TFN': TFN(input_dims, args.hidden_dim, output_dim, dropout=0.2),
        'LMF': LMF(input_dims, args.hidden_dim, output_dim, rank=4, dropout=0.2),
        'MFN': MFN(input_dims, args.hidden_dim, output_dim, memory_size=8, dropout=0.2),
        'MulT': MulT(input_dims, args.hidden_dim, output_dim, num_heads=4, num_layers=2, dropout=0.2),
        'GCN': GCNFusion(input_dims, args.hidden_dim, output_dim, num_layers=2, dropout=0.2),
        'Hypergraph': HypergraphFusion(input_dims, args.hidden_dim, output_dim, num_layers=2, dropout=0.2),
        'QuantumHybrid': QuantumHybridModel(input_dims, args.hidden_dim, output_dim, n_qubits=4, n_quantum_layers=2, dropout=0.2),
    }

    criterion = nn.BCEWithLogitsLoss()
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for name, model in models.items():
        print(f"\n训练 {name}...")
        model = model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=0.0001)
        
        best_acc, patience, best_state = 0, 0, None
        for epoch in range(args.epochs):
            train_loss, train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, is_multilabel)
            val_loss, val_metrics = validate(model, val_loader, criterion, device, is_multilabel)
            
            if val_metrics['Accuracy'] > best_acc:
                best_acc = val_metrics['Accuracy']
                best_state = model.state_dict().copy()
                patience = 0
            else:
                patience += 1
            
            if patience >= 15:
                break

        model.load_state_dict(best_state)
        _, test_metrics = validate(model, test_loader, criterion, device, is_multilabel)
        
        results[name] = {
            'accuracy': test_metrics['Accuracy'],
            'f1_micro': test_metrics.get('F1_Micro', 0),
            'f1_macro': test_metrics.get('F1_Macro', 0)
        }
        print(f"  Test Accuracy: {test_metrics['Accuracy']:.4f}")
        torch.save(best_state, save_dir / f"{name.lower()}_best.pt")

    # 保存结果
    with open(save_dir / 'all_results.json', 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*50}")
    print("结果对比:")
    print(f"{'Model':<15} {'Accuracy':>10} {'F1_Micro':>10} {'F1_Macro':>10}")
    print("-" * 50)
    for name, m in sorted(results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"{name:<15} {m['accuracy']:>10.4f} {m['f1_micro']:>10.4f} {m['f1_macro']:>10.4f}")

    print(f"\n结果保存于: {save_dir}")


if __name__ == '__main__':
    main()
