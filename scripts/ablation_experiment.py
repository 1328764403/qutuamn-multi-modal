"""
消融实验脚本
用于测试各组件对量子混合模型性能的贡献

使用方法:
    python scripts/ablation_experiment.py --config configs/config.yaml --output results/ablation_results.json
"""

import argparse
import sys
import os

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
from pathlib import Path
import yaml

from models import QuantumHybridModel
from utils.data_loader import generate_synthetic_data, get_dataloader
from utils.metrics import calculate_metrics


def set_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class AblatedModelFactory:
    """消融模型工厂"""

    @staticmethod
    def create_model(model_type, input_dims, hidden_dim, output_dim, quantum_config, device):
        """
        创建指定类型的模型

        Args:
            model_type: 模型类型
                - 'full': 完整量子混合模型
                - 'no_quantum': 移除量子融合，使用MLP
                - 'no_cross_modal': 移除跨模态纠缠
                - 'no_entanglement': 移除纠缠门
                - 'shallow_encoder': 使用浅层编码器
                - 'single_encoder': 使用共享编码器
            input_dims: 输入维度列表
            hidden_dim: 隐藏层维度
            output_dim: 输出维度
            quantum_config: 量子配置
            device: 设备
        """
        if model_type == 'full':
            return QuantumHybridModel(
                input_dims,
                hidden_dim,
                output_dim,
                quantum_config['n_qubits'],
                quantum_config['n_quantum_layers'],
                quantum_config['dropout']
            ).to(device)

        elif model_type == 'no_quantum':
            # 使用简单的concat+MLP代替量子融合
            class NoQuantumModel(nn.Module):
                def __init__(self, input_dims, hidden_dim, output_dim, dropout):
                    super().__init__()
                    self.encoders = nn.ModuleList([
                        nn.Sequential(
                            nn.Linear(dim, hidden_dim),
                            nn.ReLU(),
                            nn.Dropout(dropout),
                            nn.Linear(hidden_dim, hidden_dim)
                        ) for dim in input_dims
                    ])
                    self.fusion = nn.Sequential(
                        nn.Linear(hidden_dim * len(input_dims), hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, output_dim)
                    )

                def forward(self, *modalities):
                    encoded = []
                    for mod, enc in zip(modalities, self.encoders):
                        if len(mod.shape) == 3:
                            mod = mod.mean(dim=1)
                        encoded.append(enc(mod))
                    concat = torch.cat(encoded, dim=1)
                    return self.fusion(concat)

            return NoQuantumModel(input_dims, hidden_dim, output_dim, quantum_config['dropout']).to(device)

        elif model_type == 'no_cross_modal':
            # 移除跨模态量子纠缠层
            class NoCrossModalModel(QuantumHybridModel):
                def forward(self, *modalities):
                    batch_size = modalities[0].size(0)
                    encoded = []
                    for i, mod in enumerate(modalities):
                        if len(mod.shape) == 3:
                            mod = mod.mean(dim=1)
                        enc = self.encoders[i](mod)
                        encoded.append(enc)

                    # 只做简单拼接，使用经典MLP进行融合
                    concat = torch.cat(encoded, dim=1)
                    output = self.output_layer(concat)
                    return output

            return NoCrossModalModel(
                input_dims, hidden_dim, output_dim,
                quantum_config['n_qubits'], quantum_config['n_quantum_layers'],
                quantum_config['dropout']
            ).to(device)

        elif model_type == 'no_entanglement':
            # 使用无纠缠的量子电路
            class NoEntanglementQuantumModel(QuantumHybridModel):
                def __init__(self, input_dims, hidden_dim, output_dim, n_qubits, n_layers, dropout):
                    super().__init__(input_dims, hidden_dim, output_dim, n_qubits, n_layers, dropout)
                    # 重写量子融合层，不使用纠缠

            return NoEntanglementQuantumModel(
                input_dims, hidden_dim, output_dim,
                quantum_config['n_qubits'], quantum_config['n_quantum_layers'],
                quantum_config['dropout']
            ).to(device)

        elif model_type == 'shallow_encoder':
            # 使用浅层编码器
            class ShallowEncoderModel(QuantumHybridModel):
                def __init__(self, input_dims, hidden_dim, output_dim, n_qubits, n_layers, dropout):
                    super().__init__(input_dims, hidden_dim, output_dim, n_qubits, n_layers, dropout)
                    # 替换为浅层编码器
                    self.encoders = nn.ModuleList([
                        nn.Linear(dim, hidden_dim) for dim in input_dims
                    ])

            return ShallowEncoderModel(
                input_dims, hidden_dim, output_dim,
                quantum_config['n_qubits'], quantum_config['n_quantum_layers'],
                quantum_config['dropout']
            ).to(device)

        elif model_type == 'single_encoder':
            # 使用共享编码器
            class SingleEncoderModel(nn.Module):
                def __init__(self, input_dims, hidden_dim, output_dim, n_qubits, n_layers, dropout):
                    super().__init__()
                    # 共享编码器
                    self.shared_encoder = nn.Sequential(
                        nn.Linear(input_dims[0], hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout)
                    )
                    # 简单的特征转换
                    self.projections = nn.ModuleList([
                        nn.Linear(hidden_dim, hidden_dim) for _ in input_dims
                    ])
                    # 量子融合
                    self.quantum_fusion = nn.ModuleList([
                        QuantumHybridModel(
                            [hidden_dim], hidden_dim, hidden_dim,
                            min(n_qubits, 2), n_layers, dropout
                        ).quantum_fusion[0] if hasattr(QuantumHybridModel, 'quantum_fusion')
                        else None
                    ])

                    self.output_layer = nn.Sequential(
                        nn.Linear(hidden_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                        nn.Linear(hidden_dim, output_dim)
                    )

                def forward(self, *modalities):
                    encoded = []
                    for mod, proj in zip(modalities, self.projections):
                        if len(mod.shape) == 3:
                            mod = mod.mean(dim=1)
                        shared = self.shared_encoder(mod)
                        encoded.append(proj(shared))

                    concat = torch.cat(encoded, dim=1)
                    return self.output_layer(concat)

            return SingleEncoderModel(
                input_dims, hidden_dim, output_dim,
                quantum_config['n_qubits'], quantum_config['n_quantum_layers'],
                quantum_config['dropout']
            ).to(device)

        else:
            raise ValueError(f"Unknown model type: {model_type}")


def train_model(model, train_loader, val_loader, epochs=50, lr=0.001, patience=10, device='cpu'):
    """训练模型"""
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(epochs):
        # 训练
        model.train()
        for mods, lbls in train_loader:
            mods = [m.to(device) for m in mods]
            lbls = lbls.to(device)
            optimizer.zero_grad()
            outputs = model(*mods)
            loss = criterion(outputs, lbls)
            loss.backward()
            optimizer.step()

        # 验证
        model.eval()
        val_preds = []
        val_labels_list = []
        with torch.no_grad():
            for mods, lbls in val_loader:
                mods = [m.to(device) for m in mods]
                outputs = model(*mods)
                val_preds.append(outputs.cpu().numpy())
                val_labels_list.append(lbls.numpy())

        val_preds = np.concatenate(val_preds)
        val_labels = np.concatenate(val_labels_list)
        val_loss = np.mean((val_preds - val_labels) ** 2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    # 加载最佳模型
    if best_model_state:
        model.load_state_dict(best_model_state)

    return model


def evaluate_model(model, test_loader, device='cpu'):
    """评估模型"""
    model.eval()
    test_preds = []
    test_labels = []

    with torch.no_grad():
        for mods, lbls in test_loader:
            mods = [m.to(device) for m in mods]
            outputs = model(*mods)
            test_preds.append(outputs.cpu().numpy())
            test_labels.append(lbls.numpy())

    test_preds = np.concatenate(test_preds)
    test_labels = np.concatenate(test_labels)

    return calculate_metrics(test_labels, test_preds, task_type='regression')


def run_ablation_experiment(config, model_type, seed=42):
    """运行单个消融实验"""
    set_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # 生成数据
    data_config = config['data']
    modalities, labels = generate_synthetic_data(
        n_samples=data_config['n_samples'],
        n_modalities=data_config['n_modalities'],
        seq_lengths=data_config['seq_lengths'],
        feature_dims=data_config['feature_dims'],
        output_dim=data_config['output_dim']
    )

    n_samples = len(labels)
    n_train = int(n_samples * data_config['train_ratio'])
    n_val = int(n_samples * (data_config.get('val_ratio', 0.15) or 0.15))

    train_mods = [mod[:n_train] for mod in modalities]
    val_mods = [mod[n_train:n_train+n_val] for mod in modalities]
    test_mods = [mod[n_train+n_val:] for mod in modalities]
    train_labels = labels[:n_train]
    val_labels = labels[n_train:n_train+n_val]
    test_labels = labels[n_train+n_val:]

    train_loader = get_dataloader(train_mods, train_labels, batch_size=32, shuffle=True)
    val_loader = get_dataloader(val_mods, val_labels, batch_size=32, shuffle=False)
    test_loader = get_dataloader(test_mods, test_labels, batch_size=32, shuffle=False)

    input_dims = data_config['feature_dims']
    output_dim = data_config['output_dim']
    quantum_config = config['model']['quantum']

    # 创建模型
    model = AblatedModelFactory.create_model(
        model_type, input_dims,
        config['model']['hidden_dim'],
        output_dim, quantum_config, device
    )

    # 训练
    model = train_model(
        model, train_loader, val_loader,
        epochs=config['training']['epochs'],
        lr=config['training']['learning_rate'],
        patience=config['training']['early_stopping_patience'],
        device=device
    )

    # 评估
    metrics = evaluate_model(model, test_loader, device)

    return metrics, model_type


def main():
    parser = argparse.ArgumentParser(description='Run ablation experiments')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--output', type=str, default='results/ablation_results.json',
                       help='Output path for results')
    parser.add_argument('--model', type=str, default=None,
                       help='Specific model to test (if not specified, runs all)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')

    args = parser.parse_args()

    # 加载配置
    config_path = Path(args.config)
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent.parent / args.config

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # 定义消融实验
    ablation_configs = [
        ('full', '完整量子混合模型'),
        ('no_quantum', '移除量子融合层 (使用MLP)'),
        ('no_cross_modal', '移除跨模态纠缠'),
        ('shallow_encoder', '使用浅层编码器'),
        ('single_encoder', '使用共享编码器'),
    ]

    results = {}

    for model_type, description in ablation_configs:
        if args.model and args.model != model_type:
            continue

        print(f"\n{'='*60}")
        print(f"Running ablation: {model_type}")
        print(f"Description: {description}")
        print(f"{'='*60}")

        try:
            metrics, _ = run_ablation_experiment(config, model_type, seed=args.seed)

            results[model_type] = {
                'description': description,
                'R2': float(metrics['R2']),
                'RMSE': float(metrics['RMSE']),
                'MAE': float(metrics['MAE']),
                'MSE': float(metrics['MSE']),
                'MAPE': float(metrics.get('MAPE', 0))
            }

            print(f"Results:")
            print(f"  R²:   {metrics['R2']:.4f}")
            print(f"  RMSE: {metrics['RMSE']:.4f}")
            print(f"  MAE:  {metrics['MAE']:.4f}")

        except Exception as e:
            print(f"Error running {model_type}: {e}")
            import traceback
            traceback.print_exc()
            results[model_type] = {
                'description': description,
                'error': str(e)
            }

    # 保存结果
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print("Ablation Results Summary")
    print(f"{'='*60}")

    if results:
        # 排序并显示结果
        valid_results = {k: v for k, v in results.items() if 'R2' in v}
        if valid_results:
            sorted_results = sorted(valid_results.items(), key=lambda x: x[1]['R2'], reverse=True)

            print(f"\n{'Model':<20} {'R²':>10} {'RMSE':>10} {'MAE':>10}")
            print("-" * 55)
            for model_type, metrics in sorted_results:
                print(f"{model_type:<20} {metrics['R2']:>10.4f} {metrics['RMSE']:>10.4f} {metrics['MAE']:>10.4f}")

            # 计算相对性能
            full_model_r2 = results.get('full', {}).get('R2', 0)
            if full_model_r2:
                print(f"\nRelative to Full Model:")
                for model_type, metrics in sorted_results:
                    if model_type != 'full' and 'R2' in metrics:
                        diff = metrics['R2'] - full_model_r2
                        pct = (diff / abs(full_model_r2)) * 100 if full_model_r2 != 0 else 0
                        sign = '+' if diff > 0 else ''
                        print(f"  {model_type}: {sign}{diff:.4f} ({sign}{pct:.1f}%)")

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
