"""
查找本地FinMultiTime数据文件
"""

import os
from pathlib import Path

def find_data():
    """查找数据文件"""
    print("=" * 60)
    print("查找本地FinMultiTime数据")
    print("=" * 60)
    
    # 检查常见位置
    check_paths = [
        Path("data/finmultitime"),
        Path("quantum_multimodal_comparison/data/finmultitime"),
        Path("../data/finmultitime"),
        Path("../../data/finmultitime"),
    ]
    
    # 也检查当前目录下的所有子目录
    current_dir = Path(".")
    if current_dir.exists():
        for subdir in current_dir.iterdir():
            if subdir.is_dir() and "finmultitime" in subdir.name.lower():
                check_paths.append(subdir)
            if subdir.is_dir() and subdir.name == "data":
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir() and "finmultitime" in subsubdir.name.lower():
                        check_paths.append(subsubdir)
    
    found = False
    for data_dir in check_paths:
        if data_dir.exists() and data_dir.is_dir():
            print(f"\n✓ 找到目录: {data_dir.absolute()}")
            found = True
            
            # 列出文件
            files = list(data_dir.iterdir())
            if files:
                print(f"\n文件列表:")
                for f in sorted(files):
                    if f.is_file():
                        try:
                            size = f.stat().st_size / (1024 * 1024)
                            print(f"  - {f.name} ({size:.2f} MB)")
                        except:
                            print(f"  - {f.name}")
            
            # 检查训练/测试文件
            patterns = [
                ("SP500_train", ["SP500_train.parquet", "SP500_train.csv", "SP500_train.json", "SP500_train.jsonl"]),
                ("SP500_test", ["SP500_test.parquet", "SP500_test.csv", "SP500_test.json", "SP500_test.jsonl"]),
                ("HS300_train", ["HS300_train.parquet", "HS300_train.csv", "HS300_train.json", "HS300_train.jsonl"]),
                ("HS300_test", ["HS300_test.parquet", "HS300_test.csv", "HS300_test.json", "HS300_test.jsonl"]),
                ("train", ["train.parquet", "train.csv", "train.json", "train.jsonl"]),
                ("test", ["test.parquet", "test.csv", "test.json", "test.jsonl"]),
            ]
            
            print(f"\n检测到的数据文件:")
            for name, patterns_list in patterns:
                for pattern in patterns_list:
                    f = data_dir / pattern
                    if f.exists():
                        print(f"  ✓ {pattern}")
                        break
            
            print(f"\n建议配置:")
            print(f"  data_dir: {data_dir}")
            if any("SP500" in str(f) for f in files):
                print(f"  market: SP500")
            elif any("HS300" in str(f) for f in files):
                print(f"  market: HS300")
            else:
                print(f"  market: SP500  # 或 HS300，根据你的数据")
    
    if not found:
        print("\n✗ 未在常见位置找到数据")
        print("\n请告诉我:")
        print("1. 你的FinMultiTime数据文件在哪里？")
        print("2. 文件格式是什么？(.parquet, .csv, .json, .jsonl)")
        print("3. 文件名是什么？(例如: SP500_train.parquet 或 train.parquet)")
    
    return found


if __name__ == '__main__':
    find_data()
