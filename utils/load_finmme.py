"""
加载 FinMME 数据集并转换为多模态融合格式
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import ast
import json

from utils.offline_tokenizer import load_local_hf_tokenizer
from utils.path_guard import get_project_root, resolve_inside_project


class FinMMELoader:
    """FinMME 数据集加载器"""
    
    def __init__(
        self,
        data_dir: str = "data/finmme",
        split: str = "train",
        image_size: int = 224,
        text_max_length: int = 128,
        feature_dim: int = 768,
        use_pretrained_features: bool = True
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据分割 ('train' 或 'test')
            image_size: 图像尺寸
            text_max_length: 文本最大长度
            feature_dim: 特征维度
            use_pretrained_features: 是否使用预训练模型提取特征
        """
        # Hard guarantee: only use dataset under this project folder
        self.data_dir = resolve_inside_project(data_dir)
        self.split = split
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.feature_dim = feature_dim
        self.use_pretrained_features = use_pretrained_features
        
        # 加载数据
        self.df = self._load_data()
        
        # 初始化特征提取组件（严格离线；预训练模型仅本地可用时启用）
        self._init_models()
    
    def _load_data(self):
        """从本地加载数据文件"""
        # 优先使用 parquet，如果没有则使用 CSV
        parquet_path = self.data_dir / f"{self.split}.parquet"
        csv_path = self.data_dir / f"{self.split}.csv"
        jsonl_path = self.data_dir / f"{self.split}" / "annotations.jsonl"
        
        if parquet_path.exists():
            print(f"从本地 Parquet 加载: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            print(f"从本地 CSV 加载: {csv_path}")
            df = pd.read_csv(csv_path)
        elif jsonl_path.exists():
            print(f"从本地 JSONL 加载: {jsonl_path}")
            data = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        else:
            raise FileNotFoundError(
                f"未找到本地数据文件:\n"
                f"  - {parquet_path}\n"
                f"  - {csv_path}\n"
                f"  - {jsonl_path}\n"
                f"\n请确保数据文件存在于: {self.data_dir}\n"
                f"数据格式: train.parquet/csv 或 train/annotations.jsonl"
            )
        
        print(f"✓ 从本地加载了 {len(df)} 条数据")
        return df
    
    def _init_models(self):
        """初始化预训练模型用于特征提取"""
        print("初始化预训练模型...")
        
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 本项目要求离线可运行：只允许使用本地模型；缺失则自动降级，不联网下载
        project_root = get_project_root()
        local_bert_path = project_root / "models" / "bert-base-uncased"
        local_vit_path = project_root / "models" / "google-vit-base-patch16-224"

        # Tokenizer：本地优先，失败就用离线简易 tokenizer
        self.tokenizer = load_local_hf_tokenizer(local_bert_path)

        # 默认不加载 encoder（除非启用预训练特征且本地可用）
        self.image_encoder = None
        self.text_encoder = None
        if not self.use_pretrained_features:
            print("use_pretrained_features=false：将使用简单特征提取（离线）")
            return

        try:
            from transformers import AutoModel, ViTModel  # type: ignore
        except Exception as e:
            print(f"Warning: transformers 不可用 ({e})，将使用简单特征提取（离线）")
            self.use_pretrained_features = False
            return

        # 文本编码器（BERT）- 仅本地
        try:
            if local_bert_path.exists():
                self.text_encoder = AutoModel.from_pretrained(str(local_bert_path), local_files_only=True)
                self.text_encoder.eval()
                print(f"✓ 文本编码器: 本地 BERT ({local_bert_path})")
            else:
                print("Warning: 未找到本地 BERT，禁用预训练文本特征（不会联网下载）")
                self.text_encoder = None
        except Exception as e:
            print(f"Warning: 无法加载本地 BERT: {e}，将使用简单特征提取")
            self.text_encoder = None

        # 图像编码器（ViT）- 仅本地
        try:
            if local_vit_path.exists():
                self.image_encoder = ViTModel.from_pretrained(str(local_vit_path), local_files_only=True)
                self.image_encoder.eval()
                print(f"✓ 图像编码器: 本地 ViT ({local_vit_path})")
            else:
                print("Warning: 未找到本地 ViT，禁用预训练图像特征（不会联网下载）")
                self.image_encoder = None
        except Exception as e:
            print(f"Warning: 无法加载本地 ViT: {e}，将使用简单特征提取")
            self.image_encoder = None
    
    def _extract_image_features(self, image_path: str):
        """提取图像特征"""
        img_path = self.data_dir / image_path
        
        if not img_path.exists():
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.image_transform(img).unsqueeze(0)
            
            if self.use_pretrained_features and self.image_encoder is not None:
                # 使用 ViT 提取特征
                with torch.no_grad():
                    outputs = self.image_encoder(img_tensor)
                    features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                return features
            else:
                # 简单特征提取：展平并降维
                features = img_tensor.squeeze(0).flatten().numpy()
                # 降维到 feature_dim
                if len(features) > self.feature_dim:
                    # 使用 PCA 或简单截断
                    features = features[:self.feature_dim]
                elif len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)))
                return features.astype(np.float32)
        except Exception as e:
            print(f"Warning: 无法处理图像 {image_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _extract_text_features(self, text: str):
        """提取文本特征"""
        encoded = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.use_pretrained_features and self.text_encoder is not None:
            # 使用 BERT 提取特征
            with torch.no_grad():
                outputs = self.text_encoder(**encoded)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return features
        else:
            # 简单特征：使用 input_ids 的平均值作为特征
            features = encoded['input_ids'].float().mean(dim=1).squeeze().cpu().numpy()
            # 扩展到 feature_dim
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            elif len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            return features.astype(np.float32)
    
    def _extract_options_features(self, options):
        """提取选项特征（将选项拼接）"""
        if isinstance(options, str):
            # 尝试安全地解析字符串为列表
            try:
                # 首先尝试使用 ast.literal_eval（更安全）
                options = ast.literal_eval(options)
            except (ValueError, SyntaxError):
                try:
                    # 如果失败，尝试使用 json.loads
                    options = json.loads(options)
                except (json.JSONDecodeError, ValueError):
                    # 如果都失败，直接使用字符串
                    pass
        
        # 如果是列表，拼接所有选项
        if isinstance(options, list):
            options_text = " ".join([str(opt) for opt in options])
        else:
            options_text = str(options)
        
        return self._extract_text_features(options_text)
    
    def load_as_multimodal(self, extract_features: bool = True):
        """
        加载数据并转换为多模态格式
        
        Args:
            extract_features: 是否提取特征（True）或返回原始数据（False）
        
        Returns:
            modalities: list of numpy arrays, 每个模态的形状为 (n_samples, seq_len, feature_dim)
            labels: numpy array, 形状为 (n_samples, output_dim)
        """
        print(f"\n转换 {self.split} 数据为多模态格式...")
        
        modalities = []
        labels = []
        
        if extract_features:
            print("提取特征中...")
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="处理数据"):
                # 模态1：图像特征
                if 'image_path' in row and pd.notna(row['image_path']):
                    img_feat = self._extract_image_features(row['image_path'])
                else:
                    img_feat = np.zeros(self.feature_dim, dtype=np.float32)
                
                # 模态2：问题文本特征
                question = str(row.get('question', ''))
                text_feat = self._extract_text_features(question)
                
                # 模态3：选项特征（可选）
                options = row.get('options', [])
                if pd.notna(options) and options:
                    options_feat = self._extract_options_features(options)
                else:
                    options_feat = np.zeros(self.feature_dim, dtype=np.float32)
                
                # 添加序列维度 (seq_len=1)
                modalities.append([
                    img_feat.reshape(1, -1),      # (1, feature_dim)
                    text_feat.reshape(1, -1),    # (1, feature_dim)
                    options_feat.reshape(1, -1)  # (1, feature_dim)
                ])
                
                # 标签
                label = row.get('label', 0)
                if pd.isna(label):
                    # 如果没有 label，从 answer 推断
                    answer = row.get('answer', '')
                    options_list = row.get('options', [])
                    
                    # 安全地解析选项列表
                    if isinstance(options_list, str):
                        try:
                            options_list = ast.literal_eval(options_list)
                        except (ValueError, SyntaxError):
                            try:
                                options_list = json.loads(options_list)
                            except (json.JSONDecodeError, ValueError):
                                # 如果解析失败，尝试按行分割
                                options_list = [opt.strip() for opt in options_list.split('\n') if opt.strip()]
                    
                    # 如果 options_list 是列表，尝试找到答案的索引
                    if isinstance(options_list, list):
                        # 尝试直接匹配
                        if answer in options_list:
                            label = options_list.index(answer)
                        else:
                            # 尝试部分匹配（处理 "A: xxx" 格式）
                            for idx, opt in enumerate(options_list):
                                if isinstance(opt, str) and answer in opt:
                                    label = idx
                                    break
                            else:
                                label = 0
                    else:
                        label = 0
                
                labels.append(int(label))
        else:
            # 返回原始数据（用于调试）
            for idx, row in self.df.iterrows():
                modalities.append({
                    'image_path': row.get('image_path', None),
                    'question': row.get('question', ''),
                    'options': row.get('options', []),
                    'answer': row.get('answer', '')
                })
                label = row.get('label', 0)
                if pd.isna(label):
                    label = 0
                labels.append(int(label))
            return modalities, np.array(labels)
        
        # 转换为 numpy arrays
        n_samples = len(modalities)
        
        # 模态1：图像
        mod1 = np.stack([m[0] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态2：文本
        mod2 = np.stack([m[1] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态3：选项
        mod3 = np.stack([m[2] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        labels_array = np.array(labels).reshape(-1, 1)  # (n_samples, 1)
        
        print(f"✓ 转换完成:")
        print(f"  模态1 (图像): {mod1.shape}")
        print(f"  模态2 (文本): {mod2.shape}")
        print(f"  模态3 (选项): {mod3.shape}")
        print(f"  标签: {labels_array.shape}")
        
        return [mod1, mod2, mod3], labels_array


def load_finmme_data(
    data_dir: str = "data/finmme",
    splits: list = ["train", "test"],
    feature_dim: int = 768,
    use_pretrained_features: bool = True,
    max_samples: int = None
):
    """
    加载 FinMME 数据集
    
    Args:
        data_dir: 数据目录
        splits: 要加载的数据分割列表
        feature_dim: 特征维度
        use_pretrained_features: 是否使用预训练模型
        max_samples: 最大样本数量（用于快速测试，None表示不限制）
    
    Returns:
        dict: 包含每个分割的模态和标签
    """
    data = {}
    
    for split in splits:
        loader = FinMMELoader(
            data_dir=data_dir,
            split=split,
            feature_dim=feature_dim,
            use_pretrained_features=use_pretrained_features
        )
        modalities, labels = loader.load_as_multimodal(extract_features=True)
        
        # 限制样本数量
        if max_samples is not None and len(labels) > max_samples:
            print(f"限制 {split} 数据集为 {max_samples} 个样本（原始: {len(labels)}）")
            modalities = [mod[:max_samples] for mod in modalities]
            labels = labels[:max_samples]
        
        data[split] = {
            'modalities': modalities,
            'labels': labels
        }
    
    return data

