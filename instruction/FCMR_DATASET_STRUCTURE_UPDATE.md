# FCMR 数据集结构支持更新

## 更新说明

已更新 `utils/load_fcmr.py` 以支持 FCMR 数据集按难度级别组织的目录结构。

## 支持的数据结构

### 按难度级别组织的结构（新增支持）

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

### CSV 文件格式

`{difficulty}_data.csv` 文件应包含以下列：

- `anchor_num`: 索引号，用于定位对应的表格和文本文件
- `filename`: 图表文件名（如 `ILoBK8xwd6t8.png`）
- `correct_answer`: 答案（格式如 "1", "2, 3", "1,2,3", "None"）
- `option1`, `option2`, `option3`: 选项文本
- 其他元数据列（可选）

## 主要更新内容

### 1. `_load_data()` 方法

- 优先尝试从 `dataset/{difficulty}/{difficulty}_data.csv` 加载数据
- 支持 `difficulty="all"` 时自动加载所有难度级别的数据
- 保持向后兼容，仍支持原有的 JSON/JSONL/CSV 格式

### 2. `_extract_text_features()` 方法

- 新增 `anchor_num` 和 `difficulty` 参数
- 自动从 `{difficulty}_test_text_modality_chunk/anchor_table_test_{anchor_num}_text.txt` 加载文本数据
- 如果文件不存在，回退到使用 CSV 中的文本字段

### 3. `_extract_table_features()` 方法

- 新增 `anchor_num` 和 `difficulty` 参数
- 自动从 `{difficulty}_test_table_modality/table_modality_{anchor_num}.csv` 加载表格数据
- 将表格的数值列转换为特征向量
- 如果文件不存在，回退到使用 CSV 中的表格字段

### 4. `_extract_image_features()` 方法

- 新增 `difficulty` 参数
- 支持在多个位置查找图表文件：
  - `dataset/{difficulty}/chart_images/{filename}`
  - `dataset/{difficulty}/{filename}`
  - 以及其他可能的路径

### 5. `load_as_multimodal()` 方法

- 自动从 CSV 行中提取 `anchor_num` 和 `difficulty`
- 将参数传递给特征提取方法
- 支持 `correct_answer` 列（除了原有的 `answer` 列）
- 改进答案解析，支持 "2, 3" 这种带空格的格式

### 6. `_encode_answer()` 方法

- 改进答案解析，支持 "2, 3" 这种带空格的格式
- 自动移除空格并正确解析多选答案

## 使用方法

### 加载特定难度级别的数据

```python
from utils.load_fcmr import FCMRLoader

# 加载 easy 难度级别的数据
loader = FCMRLoader(
    data_dir="data/fcmr",
    split="test",  # 对于按难度组织的结构，split 参数会被忽略
    difficulty="easy",
    feature_dim=768
)

modalities, labels = loader.load_as_multimodal()
```

### 加载所有难度级别的数据

```python
# 加载所有难度级别的数据
loader = FCMRLoader(
    data_dir="data/fcmr",
    difficulty="all",  # 加载所有难度级别
    feature_dim=768
)

modalities, labels = loader.load_as_multimodal()
```

### 使用配置文件

在 `configs/config_fcmr.yaml` 中设置：

```yaml
data:
  source: fcmr
  data_dir: data/fcmr
  difficulty: all  # 或 "easy", "medium", "hard"
  feature_dim: 768
```

## 向后兼容性

更新后的加载器完全向后兼容原有的数据格式：

- 如果找不到按难度组织的结构，会自动回退到原有的加载方式
- 支持 JSON、JSONL、CSV、Parquet 格式
- 支持在 `data/` 子目录中查找数据文件

## 注意事项

1. **图表文件路径**: 如果图表文件不在 `chart_images` 文件夹中，加载器会在多个位置查找
2. **文件命名**: 表格和文本文件必须按照 `table_modality_{anchor_num}.csv` 和 `anchor_table_test_{anchor_num}_text.txt` 的格式命名
3. **答案格式**: 支持 "1", "2, 3", "1,2,3" 等多种格式，会自动处理空格
4. **难度级别**: 如果 CSV 文件中没有 `difficulty` 列，会自动从目录名推断

## 测试

运行以下命令测试数据加载：

```bash
cd quantum_multimodal_comparison
python -c "from utils.load_fcmr import FCMRLoader; loader = FCMRLoader(data_dir='data/fcmr', difficulty='easy'); modalities, labels = loader.load_as_multimodal(); print(f'Loaded {len(labels)} samples')"
```
