"""
检查本地FinMultiTime数据文件
帮助用户找到并验证数据文件
"""

import os
from pathlib import Path
import json

def find_finmultitime_data():
    """查找FinMultiTime数据文件"""
    print("=" * 60)
    print("查找本地FinMultiTime数据文件")
    print("=" * 60)
    
    # 可能的目录位置
    possible_dirs = [
        "data/finmultitime",
        "quantum_multimodal_comparison/data/finmultitime",
        "../data/finmultitime",
        "../../data/finmultitime",
        "D:/data/finmultitime",
        "E:/data/finmultitime",
        "C:/data/finmultitime",
    ]
    
    # 支持的文件格式
    file_patterns = [
        "*.parquet",
        "*.csv",
        "*.json",
        "*.jsonl"
    ]
    
    found_files = {}
    
    for data_dir in possible_dirs:
        dir_path = Path(data_dir)
        if dir_path.exists() and dir_path.is_dir():
            print(f"\n✓ 找到目录: {dir_path}")
            
            # 列出所有文件
            files = list(dir_path.glob("*"))
            if files:
                print(f"  文件列表:")
                for f in files[:20]:  # 只显示前20个
                    if f.is_file():
                        size = f.stat().st_size / (1024 * 1024)  # MB
                        print(f"    - {f.name} ({size:.2f} MB)")
                
                # 检查训练/测试文件
                train_files = []
                test_files = []
                
                for pattern in ["train", "test"]:
                    for ext in [".parquet", ".csv", ".json", ".jsonl"]:
                        # 带市场前缀
                        for market in ["SP500", "HS300"]:
                            f = dir_path / f"{market}_{pattern}{ext}"
                            if f.exists():
                                if pattern == "train":
                                    train_files.append(f)
                                else:
                                    test_files.append(f)
                        
                        # 不带市场前缀
                        f = dir_path / f"{pattern}{ext}"
                        if f.exists():
                            if pattern == "train":
                                train_files.append(f)
                            else:
                                test_files.append(f)
                
                if train_files:
                    print(f"\n  训练文件:")
                    for f in train_files:
                        print(f"    ✓ {f.name}")
                
                if test_files:
                    print(f"\n  测试文件:")
                    for f in test_files:
                        print(f"    ✓ {f.name}")
                
                found_files[str(dir_path)] = {
                    'train': [str(f) for f in train_files],
                    'test': [str(f) for f in test_files],
                    'all_files': [str(f) for f in files if f.is_file()]
                }
            else:
                print(f"  (目录为空)")
    
    if not found_files:
        print("\n✗ 未找到FinMultiTime数据目录")
        print("\n请检查:")
        print("1. 数据是否在以下位置之一:")
        for d in possible_dirs:
            print(f"   - {d}")
        print("\n2. 或者告诉我你的数据文件在哪里，我可以帮你配置")
    else:
        print("\n" + "=" * 60)
        print("找到的数据目录:")
        for dir_path, files in found_files.items():
            print(f"\n{dir_path}:")
            if files['train']:
                print(f"  训练文件: {len(files['train'])} 个")
            if files['test']:
                print(f"  测试文件: {len(files['test'])} 个")
    
    return found_files


def suggest_config(data_dir):
    """根据找到的数据文件建议配置"""
    print("\n" + "=" * 60)
    print("建议的配置:")
    print("=" * 60)
    
    print(f"\n在 configs/config_finmultitime.yaml 中设置:")
    print(f"data:")
    print(f"  source: finmultitime")
    print(f"  data_dir: {data_dir}")
    print(f"  market: SP500  # 或 HS300，根据你的数据文件")


if __name__ == '__main__':
    found = find_finmultitime_data()
    
    if found:
        # 使用第一个找到的目录
        first_dir = list(found.keys())[0]
        suggest_config(first_dir)
        
        print("\n" + "=" * 60)
        print("快速测试:")
        print("=" * 60)
        print(f"\n运行: python quick_test.py")
        print("然后选择选项2 (FinMultiTime)")
