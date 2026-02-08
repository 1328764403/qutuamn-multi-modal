"""
快速测试脚本 - 验证代码是否可以跑通
使用合成数据，训练少量epoch
"""

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import random

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("快速测试 - 验证代码是否可运行")
print("=" * 60)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def create_synthetic_data(n_samples=100, feature_dim=768, seq_len=1, batch_size=16):
    """创建合成多模态数据
    
    Args:
        n_samples: 样本数量
        feature_dim: 特征维度
        seq_len: 序列长度（MFN和MulT需要序列输入）
    """
    print(f"\n创建合成数据: {n_samples} 样本, seq_len={seq_len}")
    
    # 3个模态，每个模态 seq_len x feature_dim
    text_features = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
    table_features = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
    chart_features = np.random.randn(n_samples, seq_len, feature_dim).astype(np.float32)
    
    # 二分类标签：正常/异常
    labels = np.random.randint(0, 2, size=(n_samples, 1)).astype(np.float32)
    
    # 转换为tensor
    text_tensor = torch.from_numpy(text_features)
    table_tensor = torch.from_numpy(table_features)
    chart_tensor = torch.from_numpy(chart_features)
    label_tensor = torch.from_numpy(labels)
    
    dataset = TensorDataset(text_tensor, table_tensor, chart_tensor, label_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    return loader


def test_model(name, model_class, train_loader, test_loader, device, epochs=3, is_sequential=False):
    """快速训练一个模型
    
    Args:
        is_sequential: 是否需要序列输入（MFN, MulT需要）
    """
    print(f"\n测试 {name}...")
    
    input_dims = [768, 768, 768]
    
    # 不同的模型需要不同的参数
    if name == 'MulT':
        # MulT: d_model 而不是 hidden_dim
        model = model_class(input_dims, d_model=64, output_dim=1, num_heads=4, num_layers=2, dropout=0.2).to(device)
    elif name == 'MFN':
        # MFN: 使用标准参数
        model = model_class(input_dims, hidden_dim=64, output_dim=1, memory_size=8, num_layers=1, dropout=0.2).to(device)
    else:
        # TFN, LMF, GCN, Hypergraph, QuantumHybrid
        model = model_class(input_dims, hidden_dim=64, output_dim=1, dropout=0.2).to(device)
    
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 快速训练
    for epoch in range(epochs):
        model.train()
        for text, table, chart, labels in train_loader:
            text, table, chart = text.to(device), table.to(device), chart.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(text, table, chart)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        # 验证
        model.eval()
        with torch.no_grad():
            correct, total = 0, 0
            for text, table, chart, labels in test_loader:
                text, table, chart = text.to(device), table.to(device), chart.to(device)
                labels = labels.to(device)
                
                outputs = model(text, table, chart)
                preds = (torch.sigmoid(outputs) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
            acc = correct / total
        
        print(f"  Epoch {epoch+1}/{epochs} - Loss: {loss.item():.4f} - Acc: {acc:.4f}")
    
    return acc


def main():
    set_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 创建数据（MFN和MulT需要序列输入，seq_len=1也可以工作）
    train_loader = create_synthetic_data(n_samples=100, seq_len=1, batch_size=16)
    test_loader = create_synthetic_data(n_samples=30, seq_len=1, batch_size=16)
    
    # 导入模型
    from models import TFN, LMF, MFN, MulT, GCNFusion, HypergraphFusion, QuantumHybridModel
    
    models = {
        'TFN': TFN,
        'LMF': LMF,
        'MFN': MFN,
        'MulT': MulT,
        'GCN': GCNFusion,
        'Hypergraph': HypergraphFusion,
        'QuantumHybrid': QuantumHybridModel,
    }
    
    print("\n" + "=" * 60)
    print("开始测试所有模型")
    print("=" * 60)
    
    results = {}
    for name, model_class in models.items():
        try:
            acc = test_model(name, model_class, train_loader, test_loader, device, epochs=3)
            results[name] = ('✓ 成功', acc)
            print(f"  {name}: 成功! Accuracy = {acc:.4f}")
        except Exception as e:
            results[name] = ('✗ 失败', str(e))
            print(f"  {name}: 失败! 错误: {e}")
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    print(f"{'模型':<15} {'状态':<10} {'准确率':<10}")
    print("-" * 40)
    
    for name, (status, acc) in results.items():
        if isinstance(acc, float):
            print(f"{name:<15} {status:<10} {acc:.4f}")
        else:
            print(f"{name:<15} {status:<10} {'N/A':<10}")
    
    # 检查是否有失败
    failed = [name for name, (status, _) in results.items() if '失败' in status]
    if failed:
        print(f"\n⚠️ 以下模型测试失败: {failed}")
        return 1
    else:
        print("\n✅ 所有模型测试通过!")
        return 0


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
