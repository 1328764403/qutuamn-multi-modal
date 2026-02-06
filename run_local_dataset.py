"""
快速运行脚本：使用本地 FinMME 数据集训练模型
"""

import os
import sys
import subprocess
from pathlib import Path

def main():
    print("="*60)
    print("使用本地 FinMME 数据集训练模型")
    print("="*60)
    
    # 检查数据集是否存在（路径相对于 quantum_multimodal_comparison/）
    project_root = Path(__file__).resolve().parent
    data_dir = project_root / "data" / "finmme"
    required_files = [
        str(data_dir / "train.csv"),
        str(data_dir / "train.parquet"),
        str(data_dir / "test.csv"),
        str(data_dir / "test.parquet"),
    ]
    
    print("\n检查数据集文件...")
    missing_files = []
    for file in required_files:
        if os.path.exists(file):
            print(f"✓ 找到: {file}")
        else:
            print(f"✗ 缺失: {file}")
            missing_files.append(file)
    
    if missing_files:
        print(f"\n警告: 缺少以下文件:")
        for f in missing_files:
            print(f"  - {f}")
        print(f"\n请确保数据集已正确放到: {data_dir}")
        print("或者修改 configs/config_finmme.yaml 中的 data_dir 路径")
        
        response = input("\n是否继续？(y/n): ")
        if response.lower() != 'y':
            print("已取消")
            return False
    
    # 检查配置文件
    config_file = project_root / "configs" / "config_finmme.yaml"
    if not config_file.exists():
        print(f"\n错误: 配置文件不存在: {config_file}")
        return False
    
    print(f"\n✓ 使用配置文件: {config_file}")
    
    # 运行训练
    print("\n" + "="*60)
    print("开始训练模型...")
    print("="*60)
    print("\n提示:")
    print("- 本项目默认严格离线：不会联网下载预训练模型或数据集")
    print("- 若本地没有 ViT/BERT 文件，会自动降级为简单特征提取（可在配置中设 use_pretrained_features: false）")
    print("- 训练结果会保存在 results/finmme/ 目录")
    print("\n")
    
    result = subprocess.run(
        [sys.executable, str(project_root / "train.py"), "--config", str(config_file)],
        text=True
    )
    
    if result.returncode == 0:
        print("\n" + "="*60)
        print("训练完成！")
        print("="*60)
        print("\n结果保存在: results/finmme/")
        print("\n可以运行以下命令比较模型性能:")
        print("  python compare.py --results_dir results/finmme")
        return True
    else:
        print("\n训练过程中出现错误，请查看上面的错误信息")
        return False

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)






