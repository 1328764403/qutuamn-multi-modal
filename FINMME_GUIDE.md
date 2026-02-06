# FinMME 数据集使用指南

## 1. 下载数据集

首先需要下载 FinMME 数据集到本地。确保你已经登录 HuggingFace：

```bash
huggingface-cli login
```

然后运行下载脚本：

```bash
cd quantum_multimodal_comparison
python utils/download_finmme.py --output_dir data/finmme --splits train test --verify
```

这会：
1. 从 HuggingFace 下载 train 和 test 分割
2. 将图像保存到 `data/finmme/train/images/` 和 `data/finmme/test/images/`
3. 将元数据保存为 CSV 和 Parquet 格式
4. 验证下载的数据

## 2. 数据集结构

下载后的目录结构：

```
data/finmme/
├── train/
│   ├── images/
│   │   ├── train_000000.jpg
│   │   ├── train_000001.jpg
│   │   └── ...
│   ├── train.csv
│   ├── train.parquet
│   └── train_stats.json
├── test/
│   ├── images/
│   │   ├── test_000000.jpg
│   │   └── ...
│   ├── test.csv
│   ├── test.parquet
│   └── test_stats.json
```

## 3. 数据格式

FinMME 数据集包含以下字段：

- **image_path**: 图像文件路径
- **question**: 问题文本
- **options**: 选项列表（4个选项）
- **answer**: 正确答案
- **label**: 标签（0-3，对应选项索引）

## 4. 训练模型

使用 FinMME 数据集训练：

```bash
python train.py --config configs/config_finmme.yaml
```

## 5. 数据加载流程

1. **加载数据文件**：从 CSV 或 Parquet 文件读取元数据
2. **提取图像特征**：使用 ViT 模型提取图像特征（可选）
3. **提取文本特征**：使用 BERT 模型提取问题文本特征（可选）
4. **提取选项特征**：将选项拼接后提取特征
5. **转换为多模态格式**：转换为 (n_samples, seq_len, feature_dim) 格式

## 6. 配置说明

在 `configs/config_finmme.yaml` 中：

```yaml
data:
  source: finmme  # 使用 FinMME 数据集
  data_dir: data/finmme
  feature_dim: 768  # BERT/ViT 输出维度
  use_pretrained_features: true  # 使用预训练模型
```

### 选项说明

- **use_pretrained_features**: 
  - `true`: 使用 ViT 和 BERT 提取特征（推荐，但需要下载模型）
  - `false`: 使用简单的特征提取（快速，但性能较差）

## 7. 特征提取

### 使用预训练模型（推荐）

```python
from utils.load_finmme import FinMMELoader

loader = FinMMELoader(
    data_dir="data/finmme",
    split="train",
    use_pretrained_features=True
)
modalities, labels = loader.load_as_multimodal()
```

这会：
- 使用 ViT 提取图像特征（768维）
- 使用 BERT 提取文本特征（768维）
- 使用 BERT 提取选项特征（768维）

### 不使用预训练模型（快速测试）

```python
loader = FinMMELoader(
    data_dir="data/finmme",
    split="train",
    use_pretrained_features=False
)
modalities, labels = loader.load_as_multimodal()
```

这会使用简单的特征提取方法，速度更快但性能较差。

## 8. 注意事项

1. **首次运行**：需要下载预训练模型（ViT 和 BERT），可能需要一些时间
2. **内存需求**：使用预训练模型需要较多内存
3. **GPU 加速**：如果有 GPU，特征提取会更快
4. **数据路径**：确保图像路径正确，如果图像加载失败会使用零向量

## 9. 故障排除

### 问题：无法从 HuggingFace 下载

**解决方案**：
```bash
# 设置镜像（中国用户）
export HF_ENDPOINT=https://hf-mirror.com

# 或使用代理
export HF_ENDPOINT=https://hf-mirror.com
```

### 问题：预训练模型下载失败

**解决方案**：
- 检查网络连接
- 使用镜像站点
- 或设置 `use_pretrained_features: false`

### 问题：内存不足

**解决方案**：
- 减小 `batch_size`
- 设置 `use_pretrained_features: false`
- 使用更小的特征维度

## 10. 快速开始

```bash
# 1. 登录 HuggingFace
huggingface-cli login

# 2. 下载数据集
python utils/download_finmme.py --output_dir data/finmme

# 3. 训练模型
python train.py --config configs/config_finmme.yaml
```

## 11. 数据统计

下载完成后，可以查看数据统计：

```bash
python utils/download_finmme.py --output_dir data/finmme --verify
```

或直接查看 `data/finmme/train_stats.json` 和 `data/finmme/test_stats.json`。







