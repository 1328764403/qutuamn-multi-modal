# 本地数据使用指南

## 📁 数据文件格式要求

### FinMME 数据集

将数据文件放在 `data/finmme/` 目录下，支持以下格式：

**方式1: Parquet格式（推荐）**
```
data/finmme/
├── train.parquet
└── test.parquet
```

**方式2: CSV格式**
```
data/finmme/
├── train.csv
└── test.csv
```

**方式3: JSONL格式（HuggingFace格式）**
```
data/finmme/
├── train/
│   ├── annotations.jsonl
│   └── images/
│       ├── img1.jpg
│       └── ...
└── test/
    ├── annotations.jsonl
    └── images/
        └── ...
```

### FinMultiTime 数据集

将数据文件放在 `data/finmultitime/` 目录下：

**方式1: 按市场分类（推荐）**
```
data/finmultitime/
├── SP500_train.parquet
├── SP500_test.parquet
├── HS300_train.parquet
└── HS300_test.parquet
```

**方式2: 统一格式**
```
data/finmultitime/
├── train.parquet
└── test.parquet
```

**支持的格式**: `.parquet`, `.csv`, `.json`, `.jsonl`

### FCMR 数据集

将数据文件放在 `data/fcmr/` 目录下：

```
data/fcmr/
├── train.json (或 train.jsonl)
└── test.json (或 test.jsonl)
```

**支持的格式**: `.json`, `.jsonl`, `.csv`, `.parquet`

## 📥 数据下载方式

### FinMME

1. **从HuggingFace手动下载**:
   - 访问: https://huggingface.co/datasets/luojunyu/FinMME
   - 下载数据文件到 `data/finmme/`

2. **使用datasets库下载（然后保存为本地）**:
```python
from datasets import load_dataset
dataset = load_dataset("luojunyu/FinMME")
# 保存为本地文件
dataset['train'].to_parquet('data/finmme/train.parquet')
dataset['test'].to_parquet('data/finmme/test.parquet')
```

### FinMultiTime

1. **从HuggingFace手动下载**:
   - 访问: https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting
   - 下载数据文件到 `data/finmultitime/`

2. **使用datasets库下载（然后保存为本地）**:
```python
from datasets import load_dataset
dataset = load_dataset("Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting")
# 保存为本地文件
dataset['SP500']['train'].to_parquet('data/finmultitime/SP500_train.parquet')
dataset['SP500']['test'].to_parquet('data/finmultitime/SP500_test.parquet')
```

### FCMR

1. **从GitHub克隆**:
```bash
git clone https://github.com/HYU-NLP/FCMR.git
# 复制数据文件到 data/fcmr/
cp -r FCMR/data/* data/fcmr/
```

2. **手动下载**:
   - 访问: https://github.com/HYU-NLP/FCMR
   - 下载数据文件到 `data/fcmr/`

## 🔧 数据文件结构要求

### FinMME 数据列要求

CSV/Parquet文件应包含以下列：
- `image_path`: 图像文件路径
- `question`: 问题文本
- `options`: 选项（列表或JSON字符串）
- `answer`: 答案
- `label`: 标签（0-3，对应选项索引）

### FinMultiTime 数据列要求

CSV/Parquet文件应包含以下列：
- `image_path` 或 `chart_path`: K线图路径
- `news_text` 或 `text`: 新闻文本
- `table_data` 或 `financial_table`: 财务表格数据（JSON格式）
- `time_series` 或 `price_series`: 时间序列数据（列表或JSON格式）
- `close_price` 或 `label` 或 `target`: 目标变量（股价或趋势）

### FCMR 数据列要求

FCMR 数据集支持两种数据组织方式：

#### 方式1：按难度级别组织的结构（推荐）

数据目录结构：
```
data/fcmr/
├── dataset/
│   ├── easy/
│   │   ├── easy_data.csv              # 主数据文件
│   │   ├── chart_images/               # 图表图像文件夹（可选）
│   │   ├── easy_test_table_modality/   # 表格数据文件夹
│   │   │   └── table_modality_{anchor_num}.csv
│   │   └── easy_test_text_modality_chunk/  # 文本数据文件夹
│   │       └── anchor_table_test_{anchor_num}_text.txt
│   ├── medium/
│   │   ├── medium_data.csv
│   │   └── ... (同上结构)
│   └── hard/
│       ├── hard_data.csv
│       └── ... (同上结构)
```

CSV 文件应包含以下列：
- `anchor_num`: 索引号，对应表格和文本文件名
- `filename`: 图表文件名（如 `ILoBK8xwd6t8.png`）
- `correct_answer`: 答案（"1", "2, 3", "1,2,3", "None" 等）
- `option1`, `option2`, `option3`: 选项文本
- `difficulty`: 难度级别（"easy", "medium", "hard"）- 如果不存在会自动从目录名推断

#### 方式2：传统格式

JSON/JSONL文件应包含以下字段：
- `text` 或 `text_reports`: 文本报告
- `table` 或 `table_data`: 表格数据
- `chart` 或 `chart_path` 或 `image`: 图表路径
- `answer` 或 `correct_answer`: 答案（"1", "1,2", "None" 等）
- `difficulty`: 难度级别（"easy", "medium", "hard"）

## ✅ 验证数据

运行快速测试验证数据是否正确加载：

```bash
python quick_test.py
```

如果数据加载成功，会显示：
```
✓ 从本地加载了 X 条数据
```

## 🚨 常见问题

### Q: 找不到数据文件
A: 检查：
1. 文件路径是否正确
2. 文件名是否匹配（train/test）
3. 文件格式是否支持（.parquet, .csv, .json, .jsonl）

### Q: 数据格式错误
A: 确保：
1. CSV文件有正确的列名
2. JSON/JSONL文件格式正确
3. 图像路径指向正确的文件位置

### Q: 内存不足
A: 
1. 使用Parquet格式（更高效）
2. 减少 `max_samples` 限制
3. 使用 `use_pretrained_features: false`（简单特征提取）

## 📝 快速测试配置

快速测试会自动限制为100个样本，使用以下配置：

```yaml
data:
  source: finmme  # 或 finmultitime, fcmr
  data_dir: data/finmme
  max_samples: 100  # 限制100个样本
```

运行：
```bash
python quick_test.py
```
