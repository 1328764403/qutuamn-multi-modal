"""
快速下载脚本 - 将模型下载到本地 models 目录
支持镜像站点，解决网络连接问题
"""

from huggingface_hub import snapshot_download, hf_hub_download
import os
import sys
from pathlib import Path

def download_models(use_mirror: bool = True, mirror_url: str = "https://hf-mirror.com"):
    """下载特征提取器模型到本地"""
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    models_dir = project_root / "models"
    
    # 创建模型目录
    models_dir.mkdir(exist_ok=True)
    (models_dir / "bert-base-uncased").mkdir(exist_ok=True)
    (models_dir / "google-vit-base-patch16-224").mkdir(exist_ok=True)
    
    print("=" * 60)
    print("下载特征提取器模型到本地")
    print("=" * 60)
    print(f"模型将保存到: {models_dir}")
    print()
    
    # 设置镜像站点（默认使用镜像）
    if use_mirror:
        os.environ['HF_ENDPOINT'] = mirror_url
        print(f"✓ 使用镜像站点: {mirror_url}")
        print("  提示: 如果仍然失败，可以尝试:")
        print("  1. 检查网络连接")
        print("  2. 使用代理")
        print("  3. 手动从网页下载（见 models/下载说明.md）")
    else:
        print("使用官方 HuggingFace 站点")
    
    print()
    
    # 下载 BERT
    print("=" * 60)
    print("1. 下载 BERT 模型 (bert-base-uncased)")
    print("=" * 60)
    print("   这可能需要几分钟，请耐心等待...")
    try:
        snapshot_download(
            repo_id="bert-base-uncased",
            local_dir=str(models_dir / "bert-base-uncased"),
            local_dir_use_symlinks=False,
            resume_download=True  # 支持断点续传
        )
        print("✓ BERT 下载完成！")
    except Exception as e:
        print(f"✗ BERT 下载失败: {e}")
        print("\n故障排除:")
        print("1. 检查网络连接")
        print("2. 如果使用镜像，尝试直接访问: https://hf-mirror.com/models/bert-base-uncased")
        print("3. 可以手动下载文件（见 models/下载说明.md）")
        return False
    
    print()
    
    # 下载 ViT
    print("=" * 60)
    print("2. 下载 ViT 模型 (google/vit-base-patch16-224)")
    print("=" * 60)
    print("   这可能需要几分钟，请耐心等待...")
    try:
        snapshot_download(
            repo_id="google/vit-base-patch16-224",
            local_dir=str(models_dir / "google-vit-base-patch16-224"),
            local_dir_use_symlinks=False,
            resume_download=True  # 支持断点续传
        )
        print("✓ ViT 下载完成！")
    except Exception as e:
        print(f"✗ ViT 下载失败: {e}")
        print("\n故障排除:")
        print("1. 检查网络连接")
        print("2. 如果使用镜像，尝试直接访问: https://hf-mirror.com/models/google/vit-base-patch16-224")
        print("3. 可以手动下载文件（见 models/下载说明.md）")
        return False
    
    print()
    print("=" * 60)
    print("所有模型下载完成！")
    print("=" * 60)
    print(f"\n模型位置: {models_dir}")
    print("\n现在可以运行测试脚本验证:")
    print("  python utils/test_feature_extractors.py")
    
    return True

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="下载特征提取器模型")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用镜像站点（默认使用镜像）"
    )
    parser.add_argument(
        "--mirror-url",
        type=str,
        default="https://hf-mirror.com",
        help="镜像站点 URL（默认: https://hf-mirror.com）"
    )
    
    args = parser.parse_args()
    
    try:
        success = download_models(use_mirror=not args.no_mirror, mirror_url=args.mirror_url)
        if success:
            sys.exit(0)
        else:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n下载已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
