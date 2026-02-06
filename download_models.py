"""
快速下载特征提取器模型 - 自动使用镜像站点
解决网络连接超时问题
"""

from huggingface_hub import snapshot_download
import os
import sys
from pathlib import Path

# 强制使用镜像站点（解决连接超时问题）
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 项目根目录
base_dir = Path(__file__).parent
models_dir = base_dir / "models"

# 创建模型目录
models_dir.mkdir(exist_ok=True)
(models_dir / "bert-base-uncased").mkdir(exist_ok=True)
(models_dir / "google-vit-base-patch16-224").mkdir(exist_ok=True)

print("=" * 60)
print("下载特征提取器模型")
print("=" * 60)
print(f"使用镜像站点: https://hf-mirror.com")
print(f"模型保存到: {models_dir}")
print("=" * 60)
print()

# 下载 BERT
print("1. 下载 BERT 模型 (bert-base-uncased)...")
print("   文件大小约 440MB，请耐心等待...")
try:
    snapshot_download(
        repo_id="bert-base-uncased",
        local_dir=str(models_dir / "bert-base-uncased"),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("   ✓ BERT 下载完成！")
except Exception as e:
    print(f"   ✗ BERT 下载失败: {e}")
    print("\n   故障排除:")
    print("   1. 检查网络连接")
    print("   2. 尝试手动下载: https://hf-mirror.com/models/bert-base-uncased")
    print("   3. 查看详细说明: models/快速下载.md")
    sys.exit(1)

print()

# 下载 ViT
print("2. 下载 ViT 模型 (google/vit-base-patch16-224)...")
print("   文件大小约 330MB，请耐心等待...")
try:
    snapshot_download(
        repo_id="google/vit-base-patch16-224",
        local_dir=str(models_dir / "google-vit-base-patch16-224"),
        local_dir_use_symlinks=False,
        resume_download=True
    )
    print("   ✓ ViT 下载完成！")
except Exception as e:
    print(f"   ✗ ViT 下载失败: {e}")
    print("\n   故障排除:")
    print("   1. 检查网络连接")
    print("   2. 尝试手动下载: https://hf-mirror.com/models/google/vit-base-patch16-224")
    print("   3. 查看详细说明: models/快速下载.md")
    sys.exit(1)

print()
print("=" * 60)
print("✓ 所有模型下载完成！")
print("=" * 60)
print(f"\n模型位置: {models_dir}")
print("\n现在可以运行测试脚本验证:")
print("  python utils/test_feature_extractors.py")
print()
