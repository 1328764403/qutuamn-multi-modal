"""
手动下载 FinMME 数据集的辅助脚本
如果自动下载失败，可以使用此脚本手动处理已下载的文件
"""

import pandas as pd
from pathlib import Path
import json
from tqdm import tqdm
from PIL import Image
import io


def process_local_parquet(parquet_file: str, output_dir: str, split_name: str = None):
    """
    处理本地已下载的 parquet 文件
    
    Args:
        parquet_file: parquet 文件路径
        output_dir: 输出目录
        split_name: 分割名称（如果不提供，从文件名推断）
    """
    parquet_path = Path(parquet_file)
    if not parquet_path.exists():
        raise FileNotFoundError(f"文件不存在: {parquet_file}")
    
    # 推断分割名称
    if split_name is None:
        if 'train' in parquet_path.stem.lower():
            split_name = 'train'
        elif 'test' in parquet_path.stem.lower():
            split_name = 'test'
        elif 'val' in parquet_path.stem.lower() or 'dev' in parquet_path.stem.lower():
            split_name = 'validation'
        else:
            split_name = 'train'
            print(f"警告: 无法推断分割名称，使用 'train'")
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"处理文件: {parquet_file}")
    print(f"分割名称: {split_name}")
    print(f"输出目录: {output_dir}")
    print("=" * 60)
    
    # 读取 parquet
    print("读取 parquet 文件...")
    df = pd.read_parquet(parquet_file)
    print(f"加载了 {len(df)} 条数据")
    
    # 创建图像目录
    images_dir = output_path / split_name / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    
    # 处理图像
    df_export = df.copy()
    image_paths = []
    
    if 'image' in df_export.columns:
        print(f"处理图像...")
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
                img_filename = f"{split_name}_{idx:06d}.jpg"
                img_path = images_dir / img_filename
                pil_img.save(img_path, 'JPEG', quality=95)
                image_paths.append(f"{split_name}/images/{img_filename}")
            except Exception as e:
                print(f"警告: 无法处理图像 {idx}: {e}")
                image_paths.append(None)
        
        # 更新 DataFrame
        df_export['image_path'] = image_paths
        df_export = df_export.drop(columns=['image'])
    
    # 保存 CSV
    csv_path = output_path / f"{split_name}.csv"
    df_export.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"✓ CSV 已保存到: {csv_path}")
    
    # 保存为 Parquet
    parquet_output_path = output_path / f"{split_name}.parquet"
    df_export.to_parquet(parquet_output_path, index=False, compression='snappy')
    print(f"✓ Parquet 已保存到: {parquet_output_path}")
    
    # 保存统计信息
    stats = {
        'split': split_name,
        'num_samples': len(df_export),
        'columns': list(df_export.columns),
        'num_images': sum(1 for p in image_paths if p is not None) if 'image_path' in df_export.columns else 0
    }
    
    stats_path = output_path / f"{split_name}_stats.json"
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"✓ 统计信息已保存到: {stats_path}")
    
    print("\n处理完成！")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='手动处理本地 parquet 文件')
    parser.add_argument('parquet_file', type=str,
                       help='parquet 文件路径')
    parser.add_argument('--output_dir', type=str, default='data/finmme',
                       help='输出目录')
    parser.add_argument('--split', type=str, default=None,
                       help='分割名称（train/test），如果不提供会自动推断')
    
    args = parser.parse_args()
    
    process_local_parquet(
        parquet_file=args.parquet_file,
        output_dir=args.output_dir,
        split_name=args.split
    )







