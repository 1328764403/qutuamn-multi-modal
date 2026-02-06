"""
下载特征提取器模型到本地
包括 ViT (图像) 和 BERT (文本) 模型
"""

import os
import sys
from pathlib import Path
from transformers import AutoTokenizer, AutoModel, ViTModel


def download_models(use_mirror: bool = True, mirror_url: str = "https://hf-mirror.com"):
    """
    下载特征提取器模型到本地缓存
    
    Args:
        use_mirror: 是否使用镜像站点（推荐中国用户使用）
        mirror_url: 镜像站点 URL
    """
    print("=" * 60)
    print("开始下载特征提取器模型...")
    print("=" * 60)
    
    # 设置镜像站点
    if use_mirror:
        os.environ['HF_ENDPOINT'] = mirror_url
        print(f"\n使用镜像站点: {mirror_url}")
        print("提示: 如果下载失败，可以手动设置环境变量:")
        print(f"  export HF_ENDPOINT={mirror_url}")
        print("  或在 PowerShell 中:")
        print(f"  $env:HF_ENDPOINT='{mirror_url}'")
    
    models_to_download = [
        {
            "name": "BERT (文本编码器)",
            "model_id": "bert-base-uncased",
            "type": "text"
        },
        {
            "name": "ViT (图像编码器)",
            "model_id": "google/vit-base-patch16-224",
            "type": "image"
        }
    ]
    
    for model_info in models_to_download:
        model_name = model_info["name"]
        model_id = model_info["model_id"]
        model_type = model_info["type"]
        
        print(f"\n{'='*60}")
        print(f"下载 {model_name}")
        print(f"模型 ID: {model_id}")
        print(f"{'='*60}")
        
        try:
            # 下载 tokenizer（BERT 需要）
            if model_type == "text":
                print(f"\n1. 下载 Tokenizer...")
                tokenizer = AutoTokenizer.from_pretrained(model_id)
                print(f"   ✓ Tokenizer 下载完成")
                print(f"   缓存位置: {tokenizer.cache_dir if hasattr(tokenizer, 'cache_dir') else '默认缓存目录'}")
            
            # 下载模型
            print(f"\n2. 下载模型...")
            if model_type == "image":
                model = ViTModel.from_pretrained(model_id)
            else:
                model = AutoModel.from_pretrained(model_id)
            
            print(f"   ✓ 模型下载完成")
            print(f"   缓存位置: {model.cache_dir if hasattr(model, 'cache_dir') else '默认缓存目录'}")
            
            # 显示模型信息
            print(f"\n3. 模型信息:")
            print(f"   参数量: {sum(p.numel() for p in model.parameters()):,}")
            print(f"   可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
            
            print(f"\n✓ {model_name} 下载成功！")
            
        except Exception as e:
            print(f"\n✗ {model_name} 下载失败: {str(e)}")
            print(f"\n故障排除建议:")
            print(f"1. 检查网络连接")
            print(f"2. 尝试使用镜像站点（如果还没使用）")
            print(f"3. 检查 HuggingFace 访问权限")
            if use_mirror:
                print(f"4. 当前使用的镜像: {mirror_url}")
            return False
    
    print(f"\n{'='*60}")
    print("所有模型下载完成！")
    print(f"{'='*60}")
    
    # 显示缓存目录位置
    cache_dir = os.path.expanduser("~/.cache/huggingface")
    print(f"\n模型缓存目录: {cache_dir}")
    print(f"\n提示: 模型已下载到本地缓存，后续使用时会自动从缓存加载")
    
    return True


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="下载特征提取器模型到本地")
    parser.add_argument(
        "--no-mirror",
        action="store_true",
        help="不使用镜像站点（默认使用 hf-mirror.com）"
    )
    parser.add_argument(
        "--mirror-url",
        type=str,
        default="https://hf-mirror.com",
        help="镜像站点 URL（默认: https://hf-mirror.com）"
    )
    
    args = parser.parse_args()
    
    use_mirror = not args.no_mirror
    
    success = download_models(use_mirror=use_mirror, mirror_url=args.mirror_url)
    
    if success:
        print("\n✓ 所有模型下载成功！现在可以在代码中使用这些模型了。")
        sys.exit(0)
    else:
        print("\n✗ 部分模型下载失败，请检查错误信息并重试。")
        sys.exit(1)


if __name__ == "__main__":
    main()
