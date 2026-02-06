"""
下载 FinMME 数据集到本地
支持 HuggingFace 镜像站点
"""

import pandas as pd
import os
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import io
import sys


def download_finmme_dataset(
    dataset_name: str = "luojunyu/FinMME",
    output_dir: str = "data/finmme",
    splits: list = ["train", "test"],
    use_mirror: bool = True,
    mirror_url: str = "https://hf-mirror.com"
):
    """
    下载 FinMME 数据集到本地
    
    Args:
        dataset_name: HuggingFace 数据集名称
        output_dir: 输出目录
        splits: 要下载的数据分割列表
        use_mirror: 是否使用镜像站点
        mirror_url: 镜像站点 URL
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"开始下载 FinMME 数据集到 {output_dir}...")
    print("=" * 60)
    
    # 设置镜像站点
    if use_mirror:
        # 设置环境变量使用镜像
        os.environ['HF_ENDPOINT'] = mirror_url
        print(f"使用镜像站点: {mirror_url}")
        print("提示: 如果仍然失败，可以手动设置环境变量:")
        print(f"  export HF_ENDPOINT={mirror_url}")
        print("  或在 PowerShell 中:")
        print(f"  $env:HF_ENDPOINT='{mirror_url}'")
    
    for split in splits:
        print(f"\n下载 {split} 分割...")
        
        # 尝试多种方式下载
        df = None
        methods = []
        
        # 方法1: 使用 hf:// 协议（自动处理镜像）
        methods.append(("hf:// 协议", lambda: f"hf://datasets/{dataset_name}/data/{split}-00000-of-00001.parquet"))
        
        # 方法2: 使用 datasets 库
        methods.append(("datasets 库", None))
        
        # 方法3: 直接 URL（如果知道）
        if use_mirror:
            methods.append(("镜像 URL", lambda: f"{mirror_url}/datasets/{dataset_name}/resolve/main/data/{split}-00000-of-00001.parquet"))
        
        # 尝试方法1: hf:// 协议
        try:
            parquet_path = f"hf://datasets/{dataset_name}/data/{split}-00000-of-00001.parquet"
            print(f"尝试方法1: 使用 hf:// 协议读取...")
            df = pd.read_parquet(parquet_path)
            print(f"✓ 成功加载 {len(df)} 条数据")
        except Exception as e1:
            print(f"✗ 方法1 失败: {str(e1)[:100]}")
            
            # 尝试方法2: 使用 datasets 库
            try:
                print(f"尝试方法2: 使用 datasets 库...")
                from datasets import load_dataset
                
                # 设置镜像
                if use_mirror:
                    import huggingface_hub
                    huggingface_hub.constants.ENDPOINT = mirror_url
                
                hf_dataset = load_dataset(dataset_name)
                
                if split in hf_dataset:
                    split_data = hf_dataset[split]
                    df = split_data.to_pandas()
                    print(f"✓ 成功加载 {len(df)} 条数据")
                else:
                    # 尝试其他分割名称
                    available_splits = list(hf_dataset.keys())
                    print(f"可用分割: {available_splits}")
                    if available_splits:
                        split_data = hf_dataset[available_splits[0]]
                        df = split_data.to_pandas()
                        print(f"✓ 使用分割 '{available_splits[0]}' 加载 {len(df)} 条数据")
            except Exception as e2:
                print(f"✗ 方法2 失败: {str(e2)[:100]}")
                
                # 提供手动下载说明
                print("\n" + "=" * 60)
                print("自动下载失败，请尝试以下方法:")
                print("=" * 60)
                print("\n方法A: 使用镜像站点（推荐）")
                print("  1. 设置环境变量:")
                if sys.platform == 'win32':
                    print(f"     PowerShell: $env:HF_ENDPOINT='{mirror_url}'")
                    print(f"     CMD: set HF_ENDPOINT={mirror_url}")
                else:
                    print(f"     export HF_ENDPOINT={mirror_url}")
                print("  2. 重新运行此脚本")
                print("\n方法B: 手动下载")
                print(f"  1. 访问: {mirror_url}/datasets/{dataset_name}")
                print(f"  2. 下载 data/{split}-00000-of-00001.parquet")
                print(f"  3. 保存到: {output_path / f'{split}.parquet'}")
                print("\n方法C: 使用 Git LFS")
                print(f"  git lfs install")
                print(f"  git clone {mirror_url}/datasets/{dataset_name}")
                print("=" * 60)
                
                # 检查是否已有本地文件
                local_parquet = output_path / f"{split}.parquet"
                if local_parquet.exists():
                    print(f"\n发现本地文件: {local_parquet}")
                    try:
                        df = pd.read_parquet(local_parquet)
                        print(f"✓ 从本地文件加载 {len(df)} 条数据")
                    except Exception as e3:
                        print(f"✗ 本地文件读取失败: {e3}")
                        continue
                else:
                    continue
        
        if df is None:
            print(f"✗ 无法下载 {split} 分割，跳过...")
            continue
        
        # 处理下载的数据
        try:
            # 保存为 CSV（不包含图像）
            csv_path = output_path / f"{split}.csv"
            
            # 提取非图像列
            df_export = df.copy()
            
            # 处理图像：保存为文件路径或跳过
            if 'image' in df_export.columns:
                # 创建图像目录
                images_dir = output_path / split / "images"
                images_dir.mkdir(parents=True, exist_ok=True)
                
                # 保存图像并更新路径
                image_paths = []
                print(f"正在保存图像到 {images_dir}...")
                
                for idx, row in tqdm(df_export.iterrows(), total=len(df_export), desc="保存图像"):
                    img = row['image']
                    
                    # 处理不同格式的图像
                    try:
                        if isinstance(img, Image.Image):
                            pil_img = img
                        elif isinstance(img, bytes):
                            pil_img = Image.open(io.BytesIO(img))
                        elif isinstance(img, dict):
                            if 'bytes' in img:
                                pil_img = Image.open(io.BytesIO(img['bytes']))
                            elif 'path' in img:
                                pil_img = Image.open(img['path'])
                            else:
                                raise ValueError(f"Unknown image dict format: {img.keys()}")
                        else:
                            # 尝试转换为 PIL Image
                            import numpy as np
                            if isinstance(img, np.ndarray):
                                pil_img = Image.fromarray(img)
                            else:
                                pil_img = Image.fromarray(np.array(img))
                        
                        # 确保是 RGB 格式
                        if pil_img.mode != 'RGB':
                            pil_img = pil_img.convert('RGB')
                        
                        # 保存图像
                        img_filename = f"{split}_{idx:06d}.jpg"
                        img_path = images_dir / img_filename
                        pil_img.save(img_path, 'JPEG', quality=95)
                        image_paths.append(f"{split}/images/{img_filename}")
                    except Exception as img_e:
                        print(f"Warning: 无法处理图像 {idx}，跳过: {img_e}")
                        image_paths.append(None)
                
                # 更新 DataFrame
                df_export['image_path'] = image_paths
                df_export = df_export.drop(columns=['image'])
            
            # 保存 CSV
            df_export.to_csv(csv_path, index=False, encoding='utf-8')
            print(f"✓ CSV 已保存到: {csv_path}")
            
            # 保存为 Parquet（包含所有信息，但图像以路径形式）
            parquet_output_path = output_path / f"{split}.parquet"
            df_export.to_parquet(parquet_output_path, index=False, compression='snappy')
            print(f"✓ Parquet 已保存到: {parquet_output_path}")
            
            # 保存统计信息
            stats = {
                'split': split,
                'num_samples': len(df_export),
                'columns': list(df_export.columns),
                'num_images': sum(1 for p in image_paths if p is not None) if 'image_path' in df_export.columns else 0
            }
            
            stats_path = output_path / f"{split}_stats.json"
            with open(stats_path, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"✓ 统计信息已保存到: {stats_path}")
            
        except Exception as e:
            print(f"✗ 处理 {split} 分割失败: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print("\n" + "=" * 60)
    print("下载完成！")
    print(f"数据保存在: {output_dir}")
    print("=" * 60)


def verify_downloaded_data(data_dir: str = "data/finmme"):
    """验证下载的数据"""
    data_path = Path(data_dir)
    
    print("\n验证下载的数据...")
    print("=" * 60)
    
    for split in ["train", "test"]:
        csv_path = data_path / f"{split}.csv"
        parquet_path = data_path / f"{split}.parquet"
        stats_path = data_path / f"{split}_stats.json"
        
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"\n{split} 分割:")
            print(f"  CSV 文件: {csv_path}")
            print(f"  样本数: {len(df)}")
            print(f"  列: {list(df.columns)}")
            
            if 'label' in df.columns:
                print(f"  标签分布:")
                print(df['label'].value_counts().sort_index())
        else:
            print(f"\n{split} 分割: 未找到 CSV 文件")
        
        if stats_path.exists():
            with open(stats_path, 'r', encoding='utf-8') as f:
                stats = json.load(f)
            print(f"  统计信息: {stats}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='下载 FinMME 数据集')
    parser.add_argument('--output_dir', type=str, default='data/finmme',
                       help='输出目录')
    parser.add_argument('--splits', nargs='+', default=['train', 'test'],
                       help='要下载的数据分割')
    parser.add_argument('--verify', action='store_true',
                       help='下载后验证数据')
    parser.add_argument('--no-mirror', action='store_true',
                       help='不使用镜像站点')
    parser.add_argument('--mirror-url', type=str, default='https://hf-mirror.com',
                       help='镜像站点 URL')
    
    args = parser.parse_args()
    
    # 下载数据集
    download_finmme_dataset(
        output_dir=args.output_dir,
        splits=args.splits,
        use_mirror=not args.no_mirror,
        mirror_url=args.mirror_url
    )
    
    # 验证
    if args.verify:
        verify_downloaded_data(args.output_dir)

