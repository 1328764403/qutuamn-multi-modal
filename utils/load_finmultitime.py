"""
加载 FinMultiTime 数据集并转换为多模态融合格式
FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis
Paper: https://arxiv.org/html/2506.05019v1
Dataset: https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting
"""

import pandas as pd
import numpy as np
import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
from tqdm import tqdm
import json

from utils.offline_tokenizer import load_local_hf_tokenizer
from utils.path_guard import get_project_root, resolve_inside_project


class FinMultiTimeLoader:
    """FinMultiTime 数据集加载器 - 四模态金融时间序列数据集"""
    
    def __init__(
        self,
        data_dir: str = "data/finmultitime",
        split: str = "train",
        image_size: int = 224,
        text_max_length: int = 128,
        feature_dim: int = 768,
        use_pretrained_features: bool = True,
        market: str = "SP500"  # "SP500" or "HS300"
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据分割 ('train' 或 'test')
            image_size: 图像尺寸
            text_max_length: 文本最大长度
            feature_dim: 特征维度
            use_pretrained_features: 是否使用预训练模型提取特征
            market: 市场类型 ("SP500" 或 "HS300")
        """
        # Hard guarantee: only use dataset under this project folder
        self.data_dir = resolve_inside_project(data_dir)
        
        self.split = split
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.feature_dim = feature_dim
        self.use_pretrained_features = use_pretrained_features
        self.market = market
        
        # 加载数据
        self.df = self._load_data()
        
        # 初始化特征提取组件（严格离线；预训练模型仅本地可用时启用）
        self._init_models()
    
    def _load_data(self):
        """从本地加载数据文件"""
        # 确保data_dir是Path对象且已解析
        if not isinstance(self.data_dir, Path):
            self.data_dir = Path(self.data_dir)
        
        print(f"查找数据目录: {self.data_dir}")
        if not self.data_dir.exists():
            print(f"警告: 数据目录不存在: {self.data_dir}")
            print(f"当前工作目录: {Path.cwd()}")
        
        # 本地加载 - 支持多种文件格式
        parquet_path = self.data_dir / f"{self.market}_{self.split}.parquet"
        csv_path = self.data_dir / f"{self.market}_{self.split}.csv"
        json_path = self.data_dir / f"{self.market}_{self.split}.json"
        jsonl_path = self.data_dir / f"{self.market}_{self.split}.jsonl"
        
        # 也尝试不带市场前缀的文件名
        parquet_path_alt = self.data_dir / f"{self.split}.parquet"
        csv_path_alt = self.data_dir / f"{self.split}.csv"
        json_path_alt = self.data_dir / f"{self.split}.json"
        jsonl_path_alt = self.data_dir / f"{self.split}.jsonl"
        
        if parquet_path.exists():
            print(f"从本地 Parquet 加载: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif csv_path.exists():
            print(f"从本地 CSV 加载: {csv_path}")
            df = pd.read_csv(csv_path)
        elif json_path.exists():
            print(f"从本地 JSON 加载: {json_path}")
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif jsonl_path.exists():
            print(f"从本地 JSONL 加载: {jsonl_path}")
            data = []
            with open(jsonl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        elif parquet_path_alt.exists():
            print(f"从本地 Parquet 加载: {parquet_path_alt}")
            df = pd.read_parquet(parquet_path_alt)
        elif csv_path_alt.exists():
            print(f"从本地 CSV 加载: {csv_path_alt}")
            df = pd.read_csv(csv_path_alt)
        elif json_path_alt.exists():
            print(f"从本地 JSON 加载: {json_path_alt}")
            with open(json_path_alt, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif jsonl_path_alt.exists():
            print(f"从本地 JSONL 加载: {jsonl_path_alt}")
            data = []
            with open(jsonl_path_alt, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        else:
            raise FileNotFoundError(
                f"未找到本地数据文件:\n"
                f"尝试的路径:\n"
                f"  - {parquet_path}\n"
                f"  - {csv_path}\n"
                f"  - {json_path}\n"
                f"  - {jsonl_path}\n"
                f"  - {parquet_path_alt}\n"
                f"  - {csv_path_alt}\n"
                f"  - {json_path_alt}\n"
                f"  - {jsonl_path_alt}\n"
                f"\n请确保数据文件存在于: {self.data_dir}\n"
                f"支持的格式: .parquet, .csv, .json, .jsonl"
            )
        
        print(f"✓ 从本地加载了 {len(df)} 条数据 ({self.market}/{self.split})")
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
        """提取K线图特征"""
        img_path = self.data_dir / image_path if not Path(image_path).is_absolute() else Path(image_path)
        
        if not img_path.exists():
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        try:
            img = Image.open(img_path).convert('RGB')
            img_tensor = self.image_transform(img).unsqueeze(0)
            
            if self.use_pretrained_features and self.image_encoder is not None:
                with torch.no_grad():
                    outputs = self.image_encoder(img_tensor)
                    features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
                return features
            else:
                features = img_tensor.squeeze(0).flatten().numpy()
                if len(features) > self.feature_dim:
                    features = features[:self.feature_dim]
                elif len(features) < self.feature_dim:
                    features = np.pad(features, (0, self.feature_dim - len(features)))
                return features.astype(np.float32)
        except Exception as e:
            print(f"Warning: 无法处理图像 {image_path}: {e}")
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _extract_text_features(self, text: str):
        """提取新闻文本特征"""
        encoded = self.tokenizer(
            text,
            max_length=self.text_max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        if self.use_pretrained_features and self.text_encoder is not None:
            with torch.no_grad():
                outputs = self.text_encoder(**encoded)
                features = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
            return features
        else:
            features = encoded['input_ids'].float().mean(dim=1).squeeze().cpu().numpy()
            if len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            elif len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            return features.astype(np.float32)
    
    def _extract_table_features(self, table_data):
        """提取财务表格特征"""
        if isinstance(table_data, str):
            try:
                table_data = json.loads(table_data)
            except:
                table_data = {}
        
        if isinstance(table_data, dict):
            # 提取数值特征
            values = []
            for key, val in table_data.items():
                if isinstance(val, (int, float)):
                    values.append(float(val))
                elif isinstance(val, list):
                    values.extend([float(v) for v in val if isinstance(v, (int, float))])
            
            if len(values) == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # 转换为固定维度
            features = np.array(values, dtype=np.float32)
            if len(features) > self.feature_dim:
                # 降维：取均值池化或截断
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            
            return features
        else:
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def _extract_timeseries_features(self, timeseries_data):
        """提取时间序列特征"""
        if isinstance(timeseries_data, str):
            try:
                timeseries_data = json.loads(timeseries_data)
            except:
                timeseries_data = []
        
        if isinstance(timeseries_data, (list, np.ndarray)):
            features = np.array(timeseries_data, dtype=np.float32)
            if len(features) == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            # 时间序列特征：可以取统计特征（均值、方差等）或直接使用
            # 这里简化为直接使用，如果长度不匹配则截断或填充
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            
            return features
        else:
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def load_as_multimodal(self, extract_features: bool = True, target_col: str = "close_price"):
        """
        加载数据并转换为多模态格式
        
        Args:
            extract_features: 是否提取特征（True）或返回原始数据（False）
            target_col: 目标列名（用于回归任务，如预测股价）
        
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
                # 模态1：K线图（图像）
                if 'image_path' in row and pd.notna(row['image_path']):
                    img_feat = self._extract_image_features(row['image_path'])
                elif 'chart_path' in row and pd.notna(row['chart_path']):
                    img_feat = self._extract_image_features(row['chart_path'])
                else:
                    img_feat = np.zeros(self.feature_dim, dtype=np.float32)
                
                # 模态2：新闻文本
                if 'news_text' in row and pd.notna(row['news_text']):
                    text_feat = self._extract_text_features(str(row['news_text']))
                elif 'text' in row and pd.notna(row['text']):
                    text_feat = self._extract_text_features(str(row['text']))
                else:
                    text_feat = np.zeros(self.feature_dim, dtype=np.float32)
                
                # 模态3：财务表格
                if 'table_data' in row and pd.notna(row['table_data']):
                    table_feat = self._extract_table_features(row['table_data'])
                elif 'financial_table' in row and pd.notna(row['financial_table']):
                    table_feat = self._extract_table_features(row['financial_table'])
                else:
                    table_feat = np.zeros(self.feature_dim, dtype=np.float32)
                
                # 模态4：时间序列（股价）
                if 'time_series' in row and pd.notna(row['time_series']):
                    ts_feat = self._extract_timeseries_features(row['time_series'])
                elif 'price_series' in row and pd.notna(row['price_series']):
                    ts_feat = self._extract_timeseries_features(row['price_series'])
                else:
                    ts_feat = np.zeros(self.feature_dim, dtype=np.float32)
                
                # 添加序列维度 (seq_len=1)
                modalities.append([
                    img_feat.reshape(1, -1),      # (1, feature_dim)
                    text_feat.reshape(1, -1),    # (1, feature_dim)
                    table_feat.reshape(1, -1),   # (1, feature_dim)
                    ts_feat.reshape(1, -1)       # (1, feature_dim)
                ])
                
                # 标签（目标变量，如未来股价）
                if target_col in row and pd.notna(row[target_col]):
                    label = float(row[target_col])
                elif 'label' in row and pd.notna(row['label']):
                    label = float(row['label'])
                elif 'target' in row and pd.notna(row['target']):
                    label = float(row['target'])
                else:
                    label = 0.0
                
                labels.append(label)
        else:
            # 返回原始数据（用于调试）
            for idx, row in self.df.iterrows():
                modalities.append({
                    'image_path': row.get('image_path') or row.get('chart_path'),
                    'news_text': row.get('news_text') or row.get('text'),
                    'table_data': row.get('table_data') or row.get('financial_table'),
                    'time_series': row.get('time_series') or row.get('price_series')
                })
                label = row.get(target_col, row.get('label', row.get('target', 0.0)))
                labels.append(float(label))
            return modalities, np.array(labels)
        
        # 转换为 numpy arrays
        n_samples = len(modalities)
        
        # 模态1：图像
        mod1 = np.stack([m[0] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态2：文本
        mod2 = np.stack([m[1] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态3：表格
        mod3 = np.stack([m[2] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态4：时间序列
        mod4 = np.stack([m[3] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        labels_array = np.array(labels).reshape(-1, 1)  # (n_samples, 1)
        
        print(f"✓ 转换完成:")
        print(f"  模态1 (图像/K线图): {mod1.shape}")
        print(f"  模态2 (文本/新闻): {mod2.shape}")
        print(f"  模态3 (表格/财务): {mod3.shape}")
        print(f"  模态4 (时间序列/股价): {mod4.shape}")
        print(f"  标签: {labels_array.shape}")
        
        return [mod1, mod2, mod3, mod4], labels_array


def load_finmultitime_data(
    data_dir: str = "data/finmultitime",
    splits: list = ["train", "test"],
    feature_dim: int = 768,
    use_pretrained_features: bool = True,
    market: str = "SP500",
    max_samples: int = None
):
    """
    加载 FinMultiTime 数据集
    
    Args:
        data_dir: 数据目录
        splits: 要加载的数据分割列表
        feature_dim: 特征维度
        use_pretrained_features: 是否使用预训练模型
        market: 市场类型 ("SP500" 或 "HS300")
        max_samples: 最大样本数量（用于快速测试，None表示不限制）
    
    Returns:
        dict: 包含每个分割的模态和标签
    """
    data = {}
    
    for split in splits:
        loader = FinMultiTimeLoader(
            data_dir=data_dir,
            split=split,
            feature_dim=feature_dim,
            use_pretrained_features=use_pretrained_features,
            market=market
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
