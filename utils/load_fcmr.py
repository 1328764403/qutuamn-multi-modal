"""
加载 FCMR 数据集并转换为多模态融合格式
FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning
Paper: https://arxiv.org/pdf/2412.12567
Dataset: https://github.com/HYU-NLP/FCMR
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
import ast

from utils.fcmr_task_switcher import FCMRTaskSwitcher, get_fcmr_task_info
from utils.offline_tokenizer import load_local_hf_tokenizer
from utils.path_guard import get_project_root, resolve_inside_project


class FCMRLoader:
    """FCMR 数据集加载器 - 金融跨模态多跳推理基准"""
    
    def __init__(
        self,
        data_dir: str = "data/fcmr",
        split: str = "train",
        difficulty: str = "all",  # "easy", "medium", "hard", "all"
        image_size: int = 224,
        text_max_length: int = 512,  # FCMR文本较长
        feature_dim: int = 768,
        use_pretrained_features: bool = True
    ):
        """
        Args:
            data_dir: 数据目录
            split: 数据分割 ('train' 或 'test')
            difficulty: 难度级别 ('easy', 'medium', 'hard', 'all')
            image_size: 图像尺寸
            text_max_length: 文本最大长度
            feature_dim: 特征维度
            use_pretrained_features: 是否使用预训练模型提取特征
        """
        # Hard guarantee: only use dataset under this project folder
        self.data_dir = resolve_inside_project(data_dir)
        self.split = split
        self.difficulty = difficulty
        self.image_size = image_size
        self.text_max_length = text_max_length
        self.feature_dim = feature_dim
        self.use_pretrained_features = use_pretrained_features
        
        # 加载数据
        self.df = self._load_data()
        
        # 初始化模型（总是需要 tokenizer 和图像预处理，即使不使用预训练模型）
        self._init_models()
    
    def _load_data(self):
        """从本地加载数据文件"""
        # 首先尝试按难度级别组织的结构 (dataset/{difficulty}/{difficulty}_data.csv)
        if self.difficulty != "all":
            difficulty_csv_path = self.data_dir / "dataset" / self.difficulty / f"{self.difficulty}_data.csv"
            if difficulty_csv_path.exists():
                print(f"从难度级别目录加载: {difficulty_csv_path}")
                df = pd.read_csv(difficulty_csv_path)
                # 添加难度列（如果不存在）
                if 'difficulty' not in df.columns:
                    df['difficulty'] = self.difficulty
                print(f"✓ 从本地加载了 {len(df)} 条数据 (难度: {self.difficulty})")
                return df
        
        # 如果 difficulty="all"，尝试加载所有难度级别的数据
        if self.difficulty == "all":
            all_dfs = []
            for diff in ["easy", "medium", "hard"]:
                diff_csv_path = self.data_dir / "dataset" / diff / f"{diff}_data.csv"
                if diff_csv_path.exists():
                    print(f"加载难度级别: {diff}")
                    diff_df = pd.read_csv(diff_csv_path)
                    if 'difficulty' not in diff_df.columns:
                        diff_df['difficulty'] = diff
                    all_dfs.append(diff_df)
            
            if all_dfs:
                df = pd.concat(all_dfs, ignore_index=True)
                print(f"✓ 从本地加载了 {len(df)} 条数据 (所有难度级别)")
                return df
        
        # 回退到原来的加载方式
        json_path = self.data_dir / f"{self.split}.json"
        jsonl_path = self.data_dir / f"{self.split}.jsonl"
        csv_path = self.data_dir / f"{self.split}.csv"
        parquet_path = self.data_dir / f"{self.split}.parquet"
        
        # 也尝试在子目录中查找
        json_path_sub = self.data_dir / "data" / f"{self.split}.json"
        jsonl_path_sub = self.data_dir / "data" / f"{self.split}.jsonl"
        
        if json_path.exists():
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
        elif csv_path.exists():
            print(f"从本地 CSV 加载: {csv_path}")
            df = pd.read_csv(csv_path)
        elif parquet_path.exists():
            print(f"从本地 Parquet 加载: {parquet_path}")
            df = pd.read_parquet(parquet_path)
        elif json_path_sub.exists():
            print(f"从本地 JSON 加载: {json_path_sub}")
            with open(json_path_sub, 'r', encoding='utf-8') as f:
                data = json.load(f)
            df = pd.DataFrame(data)
        elif jsonl_path_sub.exists():
            print(f"从本地 JSONL 加载: {jsonl_path_sub}")
            data = []
            with open(jsonl_path_sub, 'r', encoding='utf-8') as f:
                for line in f:
                    data.append(json.loads(line))
            df = pd.DataFrame(data)
        else:
            raise FileNotFoundError(
                f"未找到本地数据文件:\n"
                f"尝试的路径:\n"
                f"  - {self.data_dir / 'dataset' / self.difficulty / f'{self.difficulty}_data.csv' if self.difficulty != 'all' else 'N/A'}\n"
                f"  - {json_path}\n"
                f"  - {jsonl_path}\n"
                f"  - {csv_path}\n"
                f"  - {parquet_path}\n"
                f"  - {json_path_sub}\n"
                f"  - {jsonl_path_sub}\n"
                f"\n请确保数据文件存在于: {self.data_dir}\n"
                f"支持的格式: .json, .jsonl, .csv, .parquet\n"
                f"或从GitHub下载: https://github.com/HYU-NLP/FCMR"
            )
        
        # 根据难度过滤
        if self.difficulty != "all" and 'difficulty' in df.columns:
            original_len = len(df)
            df = df[df['difficulty'] == self.difficulty]
            print(f"按难度 '{self.difficulty}' 过滤: {original_len} -> {len(df)} 条")
        
        print(f"✓ 从本地加载了 {len(df)} 条数据 (难度: {self.difficulty})")
        return df
    
    def _init_models(self):
        """初始化模型用于特征提取"""
        # 总是初始化图像预处理和 tokenizer（即使不使用预训练模型）
        # 图像预处理
        self.image_transform = transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # 离线 tokenizer：本地优先，失败就用简易 tokenizer（不联网）
        project_root = get_project_root()
        local_bert_path = project_root / "models" / "bert-base-uncased"
        local_vit_path = project_root / "models" / "google-vit-base-patch16-224"
        self.tokenizer = load_local_hf_tokenizer(local_bert_path)
        if not self.use_pretrained_features:
            print("不使用预训练模型，将使用简单特征提取（离线）")
            self.image_encoder = None
            self.text_encoder = None
            return

        # 只有在使用预训练特征时才加载本地预训练模型（严格离线）
        print("初始化预训练模型（仅本地，离线）...")
        try:
            from transformers import AutoModel, ViTModel  # type: ignore
        except Exception as e:
            print(f"Warning: transformers 不可用 ({e})，将使用简单特征提取（离线）")
            self.use_pretrained_features = False
            self.image_encoder = None
            self.text_encoder = None
            return

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
    
    def _extract_image_features(self, image_path: str, difficulty: str = None):
        """提取图表特征"""
        # 尝试多种路径
        img_paths = []
        
        # 1. 直接路径
        if Path(image_path).is_absolute():
            img_paths.append(Path(image_path))
        else:
            img_paths.append(self.data_dir / image_path)
        
        # 2. 在难度级别目录下的 chart_images 文件夹中查找
        if difficulty:
            img_paths.append(self.data_dir / "dataset" / difficulty / "chart_images" / image_path)
        
        # 3. 在 dataset 目录下查找
        img_paths.append(self.data_dir / "dataset" / image_path)
        
        # 4. 在所有难度级别目录下查找
        for diff in ["easy", "medium", "hard"]:
            img_paths.append(self.data_dir / "dataset" / diff / "chart_images" / image_path)
            img_paths.append(self.data_dir / "dataset" / diff / image_path)
        
        img_path = None
        for path in img_paths:
            if path.exists():
                img_path = path
                break
        
        if img_path is None or not img_path.exists():
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
    
    def _extract_text_features(self, text: str, anchor_num: int = None, difficulty: str = None):
        """提取文本报告特征"""
        # 如果提供了 anchor_num 和 difficulty，尝试从文件加载文本数据
        if anchor_num is not None and difficulty:
            text_paths = [
                self.data_dir / "dataset" / difficulty / f"{difficulty}_test_text_modality_chunk" / f"anchor_table_test_{anchor_num}_text.txt",
                self.data_dir / "dataset" / difficulty / f"anchor_table_test_{anchor_num}_text.txt",
            ]
            
            for text_path in text_paths:
                if text_path.exists():
                    try:
                        with open(text_path, 'r', encoding='utf-8') as f:
                            file_text = f.read()
                        # 如果文件内容不为空，使用文件内容；否则使用传入的 text
                        if file_text.strip():
                            text = file_text
                            break
                    except Exception as e:
                        print(f"Warning: 无法加载文本文件 {text_path}: {e}")
        
        # 如果 text 为空，返回零向量
        if not text or not text.strip():
            return np.zeros(self.feature_dim, dtype=np.float32)
        
        # 检查 tokenizer 是否已初始化
        if self.tokenizer is None:
            # 如果没有 tokenizer，使用简单的字符编码
            text_encoded = np.array([ord(c) for c in text[:self.text_max_length]], dtype=np.float32)
            if text_encoded.shape[0] < self.feature_dim:
                text_encoded = np.pad(text_encoded, (0, self.feature_dim - text_encoded.shape[0]))
            elif text_encoded.shape[0] > self.feature_dim:
                text_encoded = text_encoded[:self.feature_dim]
            return text_encoded
        
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
            # 不使用预训练模型时，使用 tokenizer 的 input_ids 作为特征
            features = encoded['input_ids'].float().mean(dim=1).squeeze().cpu().numpy()
            
            # 确保 features 是 1D 数组
            if features.ndim == 0:
                # 如果是标量，转换为数组
                features = np.array([float(features)])
            elif features.ndim > 1:
                # 如果是多维数组，展平
                features = features.flatten()
            
            # 调整特征维度
            if features.shape[0] < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - features.shape[0]))
            elif features.shape[0] > self.feature_dim:
                features = features[:self.feature_dim]
            
            return features.astype(np.float32)
    
    def _extract_table_features(self, table_data, anchor_num: int = None, difficulty: str = None):
        """提取表格特征"""
        # 如果提供了 anchor_num 和 difficulty，尝试从文件加载表格数据
        if anchor_num is not None and difficulty:
            table_paths = [
                self.data_dir / "dataset" / difficulty / f"{difficulty}_test_table_modality" / f"table_modality_{anchor_num}.csv",
                self.data_dir / "dataset" / difficulty / f"table_modality_{anchor_num}.csv",
            ]
            
            for table_path in table_paths:
                if table_path.exists():
                    try:
                        table_df = pd.read_csv(table_path)
                        # 将表格转换为数值特征
                        # 选择数值列
                        numeric_cols = table_df.select_dtypes(include=[np.number]).columns
                        if len(numeric_cols) > 0:
                            values = table_df[numeric_cols].values.flatten()
                            values = values[~np.isnan(values)]  # 移除 NaN
                            if len(values) > 0:
                                features = np.array(values, dtype=np.float32)
                                if len(features) > self.feature_dim:
                                    features = features[:self.feature_dim]
                                elif len(features) < self.feature_dim:
                                    features = np.pad(features, (0, self.feature_dim - len(features)))
                                return features
                    except Exception as e:
                        print(f"Warning: 无法加载表格文件 {table_path}: {e}")
        
        # 回退到原来的处理方式
        if isinstance(table_data, str):
            try:
                table_data = json.loads(table_data)
            except:
                try:
                    table_data = ast.literal_eval(table_data)
                except:
                    table_data = {}
        
        if isinstance(table_data, (dict, list)):
            # 提取数值特征
            values = []
            if isinstance(table_data, dict):
                for key, val in table_data.items():
                    if isinstance(val, (int, float)):
                        values.append(float(val))
                    elif isinstance(val, (list, dict)):
                        # 递归提取
                        if isinstance(val, list):
                            values.extend([float(v) for v in val if isinstance(v, (int, float))])
            elif isinstance(table_data, list):
                for item in table_data:
                    if isinstance(item, (int, float)):
                        values.append(float(item))
                    elif isinstance(item, (list, dict)):
                        if isinstance(item, list):
                            values.extend([float(v) for v in item if isinstance(v, (int, float))])
            
            if len(values) == 0:
                return np.zeros(self.feature_dim, dtype=np.float32)
            
            features = np.array(values, dtype=np.float32)
            if len(features) > self.feature_dim:
                features = features[:self.feature_dim]
            elif len(features) < self.feature_dim:
                features = np.pad(features, (0, self.feature_dim - len(features)))
            
            return features
        else:
            return np.zeros(self.feature_dim, dtype=np.float32)
    
    def load_as_multimodal(self, extract_features: bool = True, task: str = "original", task_kwargs: dict | None = None):
        """
        加载数据并转换为多模态格式
        
        Args:
            extract_features: 是否提取特征（True）或返回原始数据（False）
        
        Returns:
            modalities: list of numpy arrays, 每个模态的形状为 (n_samples, seq_len, feature_dim)
            labels: numpy array
        """
        print(f"\n转换 {self.split} 数据为多模态格式...")
        task_kwargs = task_kwargs or {}
        task_info = get_fcmr_task_info(task)
        print(f"[task] {task}: {task_info.get('description', '')}")
        
        modalities = []
        # Labels can be computed purely from df (no need to do per-row in the feature loop)
        switcher = FCMRTaskSwitcher(self.df)
        labels_array = switcher.get_labels(task, **task_kwargs)
        if task_info["task_type"] == "regression":
            labels_array = np.asarray(labels_array, dtype=np.float32).reshape(-1, 1)
        elif task_info["task_type"] == "classification" and task_info.get("is_multilabel", False):
            labels_array = np.asarray(labels_array, dtype=np.float32)
        else:
            # single-label classification: class index vector
            labels_array = np.asarray(labels_array, dtype=np.int64)
        
        if extract_features:
            print("提取特征中...")
            for idx, row in tqdm(self.df.iterrows(), total=len(self.df), desc="处理数据"):
                # 获取 anchor_num 和 difficulty
                anchor_num = None
                if 'anchor_num' in row and pd.notna(row['anchor_num']):
                    try:
                        anchor_num = int(row['anchor_num'])
                    except:
                        anchor_num = None
                
                row_difficulty = self.difficulty
                if row_difficulty == "all" and 'difficulty' in row and pd.notna(row['difficulty']):
                    row_difficulty = str(row['difficulty'])
                
                # 模态1：文本报告
                text_input = None
                if 'text' in row and pd.notna(row['text']):
                    text_input = str(row['text'])
                elif 'text_reports' in row and pd.notna(row['text_reports']):
                    text_input = str(row['text_reports'])
                
                text_feat = self._extract_text_features(
                    text_input if text_input else "", 
                    anchor_num=anchor_num, 
                    difficulty=row_difficulty if row_difficulty != "all" else None
                )
                
                # 模态2：表格
                table_input = None
                if 'table' in row and pd.notna(row['table']):
                    table_input = row['table']
                elif 'table_data' in row and pd.notna(row['table_data']):
                    table_input = row['table_data']
                
                table_feat = self._extract_table_features(
                    table_input, 
                    anchor_num=anchor_num, 
                    difficulty=row_difficulty if row_difficulty != "all" else None
                )
                
                # 模态3：图表
                chart_path = None
                if 'chart' in row and pd.notna(row['chart']):
                    chart_path = row['chart']
                elif 'chart_path' in row and pd.notna(row['chart_path']):
                    chart_path = row['chart_path']
                elif 'image' in row and pd.notna(row['image']):
                    chart_path = row['image']
                elif 'filename' in row and pd.notna(row['filename']):
                    chart_path = row['filename']
                
                chart_feat = self._extract_image_features(
                    chart_path if chart_path else "", 
                    difficulty=row_difficulty if row_difficulty != "all" else None
                )
                
                # 添加序列维度 (seq_len=1)
                modalities.append([
                    text_feat.reshape(1, -1),     # (1, feature_dim)
                    table_feat.reshape(1, -1),    # (1, feature_dim)
                    chart_feat.reshape(1, -1)    # (1, feature_dim)
                ])
        else:
            # 返回原始数据（用于调试）
            for idx, row in self.df.iterrows():
                modalities.append({
                    'text': row.get('text') or row.get('text_reports'),
                    'table': row.get('table') or row.get('table_data'),
                    'chart': row.get('chart') or row.get('chart_path') or row.get('image')
                })
            return modalities, labels_array
        
        # 转换为 numpy arrays
        n_samples = len(modalities)
        
        # 模态1：文本
        mod1 = np.stack([m[0] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态2：表格
        mod2 = np.stack([m[1] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        # 模态3：图表
        mod3 = np.stack([m[2] for m in modalities], axis=0)  # (n_samples, 1, feature_dim)
        
        print(f"✓ 转换完成:")
        print(f"  模态1 (文本报告): {mod1.shape}")
        print(f"  模态2 (表格): {mod2.shape}")
        print(f"  模态3 (图表): {mod3.shape}")
        print(f"  标签: {labels_array.shape} (task={task})")
        
        return [mod1, mod2, mod3], labels_array
    
    def _encode_answer(self, answer: str):
        """将答案编码为多标签格式"""
        # 8个可能的答案: None, 1, 2, 3, 1,2, 1,3, 2,3, 1,2,3
        answer_map = {
            'none': 0,
            '1': 1,
            '2': 2,
            '3': 3,
            '1,2': 4,
            '1,3': 5,
            '2,3': 6,
            '1,2,3': 7
        }
        
        answer_lower = answer.lower().strip()
        
        # 先尝试直接匹配
        if answer_lower in answer_map:
            idx = answer_map[answer_lower]
        else:
            # 尝试解析数字（处理 "2, 3" 这种带空格的格式）
            try:
                # 移除所有空格后分割
                nums = [int(x.strip()) for x in answer_lower.replace(' ', '').split(',') if x.strip().isdigit()]
                nums.sort()
                answer_key = ','.join(map(str, nums))
                if answer_key in answer_map:
                    idx = answer_map[answer_key]
                else:
                    idx = 0  # None
            except:
                idx = 0
        
        # 转换为one-hot编码
        label = np.zeros(8, dtype=np.float32)
        label[idx] = 1.0
        return label


def load_fcmr_data(
    data_dir: str = "data/fcmr",
    splits: list = ["train", "test"],
    difficulty: str = "all",
    feature_dim: int = 768,
    use_pretrained_features: bool = True,
    max_samples: int = None,
    task: str = "original",
    task_kwargs: dict | None = None
):
    """
    加载 FCMR 数据集
    
    Args:
        data_dir: 数据目录
        splits: 要加载的数据分割列表
        difficulty: 难度级别 ('easy', 'medium', 'hard', 'all')
        feature_dim: 特征维度
        use_pretrained_features: 是否使用预训练模型
        max_samples: 最大样本数量（用于快速测试，None表示不限制）
    
    Returns:
        dict: 包含每个分割的模态和标签
    """
    data = {}
    
    for split in splits:
        loader = FCMRLoader(
            data_dir=data_dir,
            split=split,
            difficulty=difficulty,
            feature_dim=feature_dim,
            use_pretrained_features=use_pretrained_features
        )
        modalities, labels = loader.load_as_multimodal(extract_features=True, task=task, task_kwargs=task_kwargs)
        
        # 限制样本数量
        if max_samples is not None and len(labels) > max_samples:
            print(f"限制 {split} 集样本数量为 {max_samples} (原始: {len(labels)})")
            modalities = [mod[:max_samples] for mod in modalities]
            labels = labels[:max_samples]
        
        data[split] = {
            'modalities': modalities,
            'labels': labels
        }
    
    return data
