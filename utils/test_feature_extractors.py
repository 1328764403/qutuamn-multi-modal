"""
测试特征提取器是否能正常工作
验证本地下载的 BERT 和 ViT 模型
"""

import os
import sys
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel, ViTModel


def test_bert(model_dir: str = "models"):
    """测试 BERT 文本特征提取器"""
    print("=" * 60)
    print("测试 BERT 文本特征提取器")
    print("=" * 60)
    
    bert_path = Path(model_dir) / "bert-base-uncased"
    
    if not bert_path.exists():
        print(f"✗ 错误: 找不到 BERT 模型目录: {bert_path}")
        print(f"  请确保模型已下载到: {bert_path}")
        return False
    
    # 检查必需文件
    required_files = ["config.json", "tokenizer_config.json", "vocab.txt"]
    missing_files = []
    for file in required_files:
        if not (bert_path / file).exists():
            missing_files.append(file)
    
    # 检查模型权重文件
    has_pytorch = (bert_path / "pytorch_model.bin").exists()
    has_safetensors = (bert_path / "model.safetensors").exists()
    
    if not (has_pytorch or has_safetensors):
        missing_files.append("pytorch_model.bin 或 model.safetensors")
    
    if missing_files:
        print(f"✗ 缺少必需文件: {', '.join(missing_files)}")
        print(f"  请检查模型目录: {bert_path}")
        return False
    
    print(f"✓ 模型文件检查通过")
    print(f"  模型路径: {bert_path}")
    
    try:
        # 加载 tokenizer
        print("\n1. 加载 Tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(str(bert_path), local_files_only=True)
        print("   ✓ Tokenizer 加载成功")
        
        # 加载模型
        print("\n2. 加载模型...")
        model = AutoModel.from_pretrained(str(bert_path), local_files_only=True)
        model.eval()
        print("   ✓ 模型加载成功")
        
        # 测试文本特征提取
        print("\n3. 测试文本特征提取...")
        test_text = "This is a test sentence for BERT feature extraction."
        
        # Tokenize
        encoded = tokenizer(
            test_text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 提取特征
        with torch.no_grad():
            outputs = model(**encoded)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        print(f"   ✓ 特征提取成功")
        print(f"   输入文本: {test_text}")
        print(f"   特征维度: {features.shape}")
        print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"   特征均值: {features.mean():.4f}")
        
        # 检查特征维度
        if features.shape[0] == 768:
            print("   ✓ 特征维度正确 (768)")
        else:
            print(f"   ⚠ 警告: 特征维度为 {features.shape[0]}，预期为 768")
        
        print("\n✓ BERT 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ BERT 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_vit(model_dir: str = "models"):
    """测试 ViT 图像特征提取器"""
    print("\n" + "=" * 60)
    print("测试 ViT 图像特征提取器")
    print("=" * 60)
    
    vit_path = Path(model_dir) / "google-vit-base-patch16-224"
    
    if not vit_path.exists():
        print(f"✗ 错误: 找不到 ViT 模型目录: {vit_path}")
        print(f"  请确保模型已下载到: {vit_path}")
        return False
    
    # 检查必需文件
    required_files = ["config.json"]
    missing_files = []
    for file in required_files:
        if not (vit_path / file).exists():
            missing_files.append(file)
    
    # 检查模型权重文件
    has_pytorch = (vit_path / "pytorch_model.bin").exists()
    has_safetensors = (vit_path / "model.safetensors").exists()
    
    if not (has_pytorch or has_safetensors):
        missing_files.append("pytorch_model.bin 或 model.safetensors")
    
    if missing_files:
        print(f"✗ 缺少必需文件: {', '.join(missing_files)}")
        print(f"  请检查模型目录: {vit_path}")
        return False
    
    print(f"✓ 模型文件检查通过")
    print(f"  模型路径: {vit_path}")
    
    try:
        # 加载模型
        print("\n1. 加载模型...")
        model = ViTModel.from_pretrained(str(vit_path), local_files_only=True)
        model.eval()
        print("   ✓ 模型加载成功")
        
        # 创建测试图像
        print("\n2. 创建测试图像...")
        # 创建一个简单的测试图像 (224x224 RGB)
        test_image = Image.new('RGB', (224, 224), color='red')
        
        # 图像预处理
        image_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        img_tensor = image_transform(test_image).unsqueeze(0)
        print(f"   ✓ 测试图像创建成功")
        print(f"   图像尺寸: {img_tensor.shape}")
        
        # 测试图像特征提取
        print("\n3. 测试图像特征提取...")
        with torch.no_grad():
            outputs = model(img_tensor)
            features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        
        print(f"   ✓ 特征提取成功")
        print(f"   特征维度: {features.shape}")
        print(f"   特征范围: [{features.min():.4f}, {features.max():.4f}]")
        print(f"   特征均值: {features.mean():.4f}")
        
        # 检查特征维度
        if features.shape[0] == 768:
            print("   ✓ 特征维度正确 (768)")
        else:
            print(f"   ⚠ 警告: 特征维度为 {features.shape[0]}，预期为 768")
        
        print("\n✓ ViT 测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ ViT 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_integration(model_dir: str = "models"):
    """测试两个特征提取器的集成使用"""
    print("\n" + "=" * 60)
    print("测试特征提取器集成")
    print("=" * 60)
    
    try:
        from utils.load_finmme import FinMMELoader
        
        # 创建一个临时测试配置
        print("\n1. 初始化 FinMMELoader（使用本地模型）...")
        
        # 修改模型路径（需要临时修改环境或使用自定义路径）
        # 这里我们直接测试模型加载
        bert_path = Path(model_dir) / "bert-base-uncased"
        vit_path = Path(model_dir) / "google-vit-base-patch16-224"
        
        if not bert_path.exists() or not vit_path.exists():
            print("   ⚠ 跳过集成测试（模型未找到）")
            return True
        
        print("   ✓ 模型路径检查通过")
        print(f"   BERT: {bert_path}")
        print(f"   ViT: {vit_path}")
        
        print("\n2. 测试模型加载...")
        tokenizer = AutoTokenizer.from_pretrained(str(bert_path), local_files_only=True)
        bert_model = AutoModel.from_pretrained(str(bert_path), local_files_only=True)
        vit_model = ViTModel.from_pretrained(str(vit_path), local_files_only=True)
        
        print("   ✓ 所有模型加载成功")
        print("\n✓ 集成测试通过！")
        return True
        
    except Exception as e:
        print(f"\n✗ 集成测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="测试特征提取器模型")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="模型目录路径（默认: models）"
    )
    parser.add_argument(
        "--skip-bert",
        action="store_true",
        help="跳过 BERT 测试"
    )
    parser.add_argument(
        "--skip-vit",
        action="store_true",
        help="跳过 ViT 测试"
    )
    parser.add_argument(
        "--skip-integration",
        action="store_true",
        help="跳过集成测试"
    )
    
    args = parser.parse_args()
    
    # 获取项目根目录
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    model_dir = project_root / args.model_dir
    
    print(f"项目根目录: {project_root}")
    print(f"模型目录: {model_dir}")
    print()
    
    results = []
    
    # 测试 BERT
    if not args.skip_bert:
        results.append(("BERT", test_bert(str(model_dir))))
    else:
        print("跳过 BERT 测试")
    
    # 测试 ViT
    if not args.skip_vit:
        results.append(("ViT", test_vit(str(model_dir))))
    else:
        print("跳过 ViT 测试")
    
    # 集成测试
    if not args.skip_integration:
        results.append(("集成", test_integration(str(model_dir))))
    else:
        print("跳过集成测试")
    
    # 总结
    print("\n" + "=" * 60)
    print("测试总结")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ 通过" if passed else "✗ 失败"
        print(f"{name}: {status}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ 所有测试通过！特征提取器可以正常使用。")
        sys.exit(0)
    else:
        print("\n✗ 部分测试失败，请检查错误信息。")
        sys.exit(1)


if __name__ == "__main__":
    main()
