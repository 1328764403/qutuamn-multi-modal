"""
快速测试脚本 - 验证代码和模型是否正常工作
使用100个真实数据样本快速运行
"""

import torch
import yaml
import argparse
from pathlib import Path
import sys

# 修改train.py以支持max_samples限制
def quick_test():
    """快速测试"""
    print("=" * 60)
    print("快速测试模式 - 使用100个真实数据样本验证代码")
    print("=" * 60)
    
    # 选择数据集
    print("\n选择要使用的数据集:")
    print("1. FinMME (推荐，数据较小，下载快)")
    print("2. FinMultiTime (数据较大，需要从HuggingFace下载)")
    
    choice = input("\n请输入选择 (1/2，默认1): ").strip() or "1"
    
    if choice == "2":
        config_path = "configs/config_quick_test_finmultitime.yaml"
        dataset_name = "FinMultiTime"
    else:
        config_path = "configs/config_quick_test.yaml"
        dataset_name = "FinMME"
    
    if not Path(config_path).exists():
        print(f"错误: 配置文件不存在: {config_path}")
        return
    
    print(f"\n使用配置文件: {config_path}")
    print(f"数据集: {dataset_name}")
    print("\n这将:")
    print("- 使用真实数据集（限制100个样本）")
    print("- 训练3个模型（TFN, LMF, QuantumHybrid）")
    print("- 只训练5个epoch")
    print("- 使用小模型规模")
    print("\n注意: 首次运行需要下载/加载数据集，可能需要一些时间")
    
    input("\n按Enter继续，或Ctrl+C取消...")
    
    # 运行训练
    sys.argv = ['train.py', '--config', config_path]
    
    try:
        # 导入并运行训练主函数
        from train import main as train_main
        train_main()
        print("\n" + "=" * 60)
        print("✓ 快速测试完成！")
        print("=" * 60)
        print("\n如果测试成功，可以:")
        print("1. 运行完整实验: python train.py --config configs/config_finmultitime.yaml")
        print("2. 生成论文: python generate_paper.py")
    except Exception as e:
        print(f"\n✗ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    quick_test()
