"""
FCMR 数据集加载测试脚本
仅测试数据加载功能，不训练模型
"""

import sys
from pathlib import Path
import numpy as np

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

from utils.load_fcmr import FCMRLoader


def test_data_loading():
    """测试 FCMR 数据加载"""
    print("=" * 70)
    print("FCMR 数据集加载测试")
    print("=" * 70)
    
    data_dir = "data/fcmr"
    
    # 检查数据目录
    if not Path(data_dir).exists():
        print(f"\n错误: 数据目录不存在: {data_dir}")
        print("请确保 FCMR 数据集已下载并放置在正确的位置")
        return False
    
    # 测试不同难度级别
    difficulties = ["easy", "medium", "hard"]
    max_samples = 10  # 每个难度级别只加载10个样本进行测试
    
    for difficulty in difficulties:
        print(f"\n{'='*70}")
        print(f"测试难度级别: {difficulty}")
        print(f"{'='*70}")
        
        try:
            # 创建加载器
            loader = FCMRLoader(
                data_dir=data_dir,
                split="train",  # 对于按难度组织的结构，split 参数会被忽略
                difficulty=difficulty,
                feature_dim=768,
                use_pretrained_features=False  # 快速测试不使用预训练模型
            )
            
            print(f"✓ 加载器创建成功")
            print(f"  数据框大小: {len(loader.df)} 行")
            
            # 显示前几行数据
            if len(loader.df) > 0:
                print(f"\n数据列: {list(loader.df.columns)}")
                print(f"\n前3行数据预览:")
                print(loader.df.head(3).to_string())
            
            # 加载多模态数据（限制样本数）
            print(f"\n加载多模态数据（限制为 {max_samples} 个样本）...")
            modalities, labels = loader.load_as_multimodal(extract_features=True)
            
            # 限制样本数
            if len(labels) > max_samples:
                modalities = [mod[:max_samples] for mod in modalities]
                labels = labels[:max_samples]
            
            print(f"✓ 数据加载成功")
            print(f"  样本数: {len(labels)}")
            print(f"  模态1 (文本): {modalities[0].shape}")
            print(f"  模态2 (表格): {modalities[1].shape}")
            print(f"  模态3 (图表): {modalities[2].shape}")
            print(f"  标签: {labels.shape}")
            
            # 检查数据质量
            print(f"\n数据质量检查:")
            
            # 检查是否有 NaN
            has_nan = False
            for i, mod in enumerate(modalities):
                nan_count = np.isnan(mod).sum()
                if nan_count > 0:
                    print(f"  ⚠ 模态{i+1} 包含 {nan_count} 个 NaN 值")
                    has_nan = True
                else:
                    print(f"  ✓ 模态{i+1} 无 NaN 值")
            
            if has_nan:
                print(f"  ⚠ 警告: 数据中包含 NaN 值，可能需要处理")
            
            # 检查标签分布
            if len(labels.shape) == 2:
                # 多标签分类
                label_counts = labels.sum(axis=0)
                print(f"\n标签分布 (8个类别):")
                for i, count in enumerate(label_counts):
                    print(f"  类别 {i}: {int(count)} 个样本")
            else:
                # 单标签
                unique_labels, counts = np.unique(labels, return_counts=True)
                print(f"\n标签分布:")
                for label, count in zip(unique_labels, counts):
                    print(f"  标签 {label}: {count} 个样本")
            
            print(f"\n✓ {difficulty} 难度级别测试通过")
            
        except FileNotFoundError as e:
            print(f"\n✗ 文件未找到: {e}")
            print(f"  请确保 {data_dir}/dataset/{difficulty}/{difficulty}_data.csv 存在")
            continue
        except Exception as e:
            print(f"\n✗ {difficulty} 难度级别测试失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # 测试加载所有难度级别
    print(f"\n{'='*70}")
    print(f"测试加载所有难度级别")
    print(f"{'='*70}")
    
    try:
        loader = FCMRLoader(
            data_dir=data_dir,
            split="train",
            difficulty="all",
            feature_dim=768,
            use_pretrained_features=False
        )
        
        print(f"✓ 加载器创建成功")
        print(f"  数据框大小: {len(loader.df)} 行")
        
        # 限制样本数
        if len(loader.df) > max_samples:
            loader.df = loader.df.head(max_samples)
            print(f"  限制为 {max_samples} 个样本进行测试")
        
        modalities, labels = loader.load_as_multimodal(extract_features=True)
        
        print(f"✓ 所有难度级别数据加载成功")
        print(f"  样本数: {len(labels)}")
        print(f"  模态1 (文本): {modalities[0].shape}")
        print(f"  模态2 (表格): {modalities[1].shape}")
        print(f"  模态3 (图表): {modalities[2].shape}")
        print(f"  标签: {labels.shape}")
        
        print(f"\n✓ 所有难度级别测试通过")
        
    except Exception as e:
        print(f"\n✗ 所有难度级别测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    print("\n" + "=" * 70)
    print("✓ 所有数据加载测试完成！")
    print("=" * 70)
    print("\n如果所有测试通过，可以运行完整测试:")
    print("  python test_fcmr_quick.py")
    print("\n或运行完整训练:")
    print("  python train.py --config configs/config_fcmr_quick_test.yaml")
    
    return True


if __name__ == '__main__':
    success = test_data_loading()
    sys.exit(0 if success else 1)
