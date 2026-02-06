"""
FCMR 数据集快速测试脚本
使用真实小样本数据快速验证代码和模型
"""

import torch
import torch.nn as nn
import yaml
import argparse
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from models import TFN, LMF, MFN, MulT, GCNFusion, HypergraphFusion, QuantumHybridModel
from utils.data_loader import get_dataloader
from utils.load_fcmr import load_fcmr_data
from utils.metrics import calculate_metrics


def test_model_forward(model, dataloader, device):
    """测试模型前向传播"""
    model.eval()
    total_samples = 0
    successful_forward = 0
    
    with torch.no_grad():
        for modalities, labels in dataloader:
            try:
                modalities = [mod.to(device) for mod in modalities]
                labels = labels.to(device)
                
                outputs = model(*modalities)
                
                # 检查输出形状
                if outputs.shape[0] == labels.shape[0]:
                    successful_forward += labels.shape[0]
                total_samples += labels.shape[0]
            except Exception as e:
                print(f"前向传播错误: {e}")
                return False
    
    print(f"✓ 前向传播测试: {successful_forward}/{total_samples} 样本成功")
    return successful_forward == total_samples


def quick_test_fcmr():
    """快速测试 FCMR 数据集"""
    print("=" * 70)
    print("FCMR 数据集快速测试")
    print("=" * 70)
    
    # 检查配置文件
    config_path = "configs/config_fcmr_quick_test.yaml"
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        print("请确保配置文件存在")
        return
    
    # 加载配置
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    print(f"\n使用配置文件: {config_path}")
    print(f"数据目录: {config['data']['data_dir']}")
    print(f"难度级别: {config['data']['difficulty']}")
    print(f"最大样本数: {config['data'].get('max_samples', '无限制')}")
    print(f"批次大小: {config['data']['batch_size']}")
    print(f"训练轮数: {config['training']['epochs']}")
    print(f"要训练的模型: {config['models_to_train']}")
    
    # 检查数据目录
    data_dir = Path(config['data']['data_dir'])
    if not data_dir.exists():
        print(f"\n错误: 数据目录不存在: {data_dir}")
        print("请确保 FCMR 数据集已下载并放置在正确的位置")
        return
    
    # 检查是否有数据文件
    dataset_dir = data_dir / "dataset"
    if not dataset_dir.exists():
        print(f"\n警告: 未找到 dataset 目录: {dataset_dir}")
        print("将尝试使用其他数据格式")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n使用设备: {device}")
    
    # 加载数据
    print("\n" + "=" * 70)
    print("步骤 1: 加载数据")
    print("=" * 70)
    
    try:
        difficulty = config['data'].get('difficulty', 'all')
        max_samples = config['data'].get('max_samples', None)
        
        print(f"正在加载 FCMR 数据 (难度: {difficulty}, 最大样本数: {max_samples})...")
        
        # 注意：FCMR 数据集按难度组织，没有 train/test 分割
        # 我们使用所有数据，然后手动分割
        # 使用 FCMRLoader 直接加载数据
        from utils.load_fcmr import FCMRLoader
        
        loader = FCMRLoader(
            data_dir=config['data']['data_dir'],
            split="train",  # 对于按难度组织的结构，split 参数会被忽略
            difficulty=difficulty,
            feature_dim=config['data'].get('feature_dim', 768),
            use_pretrained_features=config['data'].get('use_pretrained_features', True)
        )
        
        all_mods, all_labels = loader.load_as_multimodal(extract_features=True)
        
        # 限制样本数量
        if max_samples is not None and len(all_labels) > max_samples:
            print(f"限制样本数量为 {max_samples} (原始: {len(all_labels)})")
            all_mods = [mod[:max_samples] for mod in all_mods]
            all_labels = all_labels[:max_samples]
        
        print(f"✓ 成功加载 {len(all_labels)} 个样本")
        print(f"  模态1 (文本): {all_mods[0].shape}")
        print(f"  模态2 (表格): {all_mods[1].shape}")
        print(f"  模态3 (图表): {all_mods[2].shape}")
        print(f"  标签: {all_labels.shape}")
        
        # 分割数据
        n_samples = len(all_labels)
        train_ratio = config['data']['train_ratio']
        val_ratio = config['data']['val_ratio']
        test_ratio = config['data']['test_ratio']
        
        n_train = int(n_samples * train_ratio)
        n_val = int(n_samples * val_ratio)
        n_test = n_samples - n_train - n_val
        
        # 确保每个集合至少有一个样本
        if n_val == 0:
            n_val = 1
            n_train = n_samples - n_val - n_test
        if n_test == 0:
            n_test = 1
            n_train = n_samples - n_val - n_test
        
        train_mods = [mod[:n_train] for mod in all_mods]
        val_mods = [mod[n_train:n_train+n_val] for mod in all_mods]
        test_mods = [mod[n_train+n_val:] for mod in all_mods]
        
        train_labels = all_labels[:n_train]
        val_labels = all_labels[n_train:n_train+n_val]
        test_labels = all_labels[n_train+n_val:]
        
        print(f"\n数据分割:")
        print(f"  训练集: {len(train_labels)} 个样本")
        print(f"  验证集: {len(val_labels)} 个样本")
        print(f"  测试集: {len(test_labels)} 个样本")
        
        # 更新配置
        config['data']['n_modalities'] = len(train_mods)
        config['data']['feature_dims'] = [mod.shape[-1] for mod in train_mods]
        config['data']['seq_lengths'] = [mod.shape[1] for mod in train_mods]
        config['data']['output_dim'] = train_labels.shape[1] if len(train_labels.shape) > 1 else 8
        
    except Exception as e:
        print(f"\n✗ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 创建数据加载器
    print("\n" + "=" * 70)
    print("步骤 2: 创建数据加载器")
    print("=" * 70)
    
    try:
        batch_size = config['data']['batch_size']
        train_loader = get_dataloader(
            train_mods, train_labels, 
            batch_size=batch_size, 
            shuffle=True,
            seq_lengths=config['data']['seq_lengths']
        )
        val_loader = get_dataloader(
            val_mods, val_labels,
            batch_size=batch_size,
            shuffle=False,
            seq_lengths=config['data']['seq_lengths']
        )
        test_loader = get_dataloader(
            test_mods, test_labels,
            batch_size=batch_size,
            shuffle=False,
            seq_lengths=config['data']['seq_lengths']
        )
        
        print(f"✓ 数据加载器创建成功")
        print(f"  训练批次: {len(train_loader)}")
        print(f"  验证批次: {len(val_loader)}")
        print(f"  测试批次: {len(test_loader)}")
        
    except Exception as e:
        print(f"\n✗ 数据加载器创建失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试模型
    print("\n" + "=" * 70)
    print("步骤 3: 测试模型")
    print("=" * 70)
    
    # 获取输入维度（feature_dims）
    input_dims = config['data']['feature_dims']
    output_dim = config['data']['output_dim']
    
    # 模型配置（与 train.py 保持一致）
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
    
    models_to_test = config['models_to_train']
    results = {}
    
    for model_name in models_to_test:
        if model_name not in model_configs:
            print(f"\n⚠ 跳过未知模型: {model_name}")
            continue
        
        print(f"\n{'='*70}")
        print(f"测试模型: {model_name}")
        print(f"{'='*70}")
        
        try:
            model_config = model_configs[model_name]
            ModelClass = model_config['class']
            model_args = model_config['args']
            
            # 创建模型
            model = ModelClass(**model_args).to(device)
            
            print(f"✓ 模型创建成功")
            print(f"  参数数量: {sum(p.numel() for p in model.parameters()):,}")
            
            # 测试前向传播
            print(f"\n测试前向传播...")
            if test_model_forward(model, train_loader, device):
                print(f"✓ {model_name} 前向传播测试通过")
                results[model_name] = "通过"
            else:
                print(f"✗ {model_name} 前向传播测试失败")
                results[model_name] = "失败"
            
        except Exception as e:
            print(f"\n✗ {model_name} 测试失败: {e}")
            import traceback
            traceback.print_exc()
            results[model_name] = f"错误: {str(e)}"
    
    # 总结
    print("\n" + "=" * 70)
    print("测试总结")
    print("=" * 70)
    
    for model_name, result in results.items():
        status = "✓" if result == "通过" else "✗"
        print(f"{status} {model_name}: {result}")
    
    print("\n" + "=" * 70)
    print("快速测试完成！")
    print("=" * 70)
    print("\n如果所有模型测试通过，可以运行完整训练:")
    print("  python train.py --config configs/config_fcmr.yaml")


if __name__ == '__main__':
    quick_test_fcmr()
