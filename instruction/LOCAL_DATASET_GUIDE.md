# 本地数据集使用指南

## 快速开始

如果您已经将数据集下载到本地，可以直接按照以下步骤运行：

### 1. 确认数据集结构

确保您的数据集目录结构如下：

```
data/finmme/
├── train.csv          # 训练集元数据
├── train.parquet      # 训练集元数据（Parquet格式）
├── test.csv           # 测试集元数据
├── test.parquet       # 测试集元数据（Parquet格式）
├── train/
│   └── images/        # 训练集图像文件
│       ├── train_000000.jpg
│       ├── train_000001.jpg
│       └── ...
└── test/
    └── images/        # 测试集图像文件
        ├── test_000000.jpg
        └── ...
```

### 2. 使用配置文件运行

项目已经提供了 `configs/config_finmme.yaml` 配置文件，默认数据路径为 `data/finmme`。

**直接运行训练：**

```bash
python train.py --config configs/config_finmme.yaml
```

### 3. 如果数据集在其他位置

如果您的数据集不在 `data/finmme` 目录，有两种方法：

#### 方法1：修改配置文件

编辑 `configs/config_finmme.yaml`，修改 `data_dir` 字段：

```yaml
data:
  source: finmme
  data_dir: /path/to/your/dataset  # 修改为您的数据集路径
  ...
```

#### 方法2：使用命令行参数（如果支持）

或者创建一个新的配置文件，例如 `configs/config_custom.yaml`：

```yaml
# 复制 config_finmme.yaml 的内容，然后修改 data_dir
data:
  source: finmme
  data_dir: /path/to/your/dataset
  ...
```

然后运行：
```bash
python train.py --config configs/config_custom.yaml
```

### 4. 运行完整流程

如果您想运行完整的训练和比较流程：

```bash
# 方式1：使用 run_all.py（但需要修改为使用 finmme 配置）
# 需要先修改 run_all.py 中的配置文件路径

# 方式2：手动运行各个步骤
python train.py --config configs/config_finmme.yaml
python compare.py --results_dir results/finmme
```

### 5. 配置选项说明

在 `configs/config_finmme.yaml` 中，重要的配置项：

```yaml
data:
  source: finmme              # 数据源：'synthetic' 或 'finmme'
  data_dir: data/finmme       # 数据集目录路径
  feature_dim: 768            # 特征维度（BERT/ViT输出维度）
  use_pretrained_features: true  # 是否使用预训练模型提取特征
  
training:
  epochs: 30                  # 训练轮数
  learning_rate: 0.0001       # 学习率
  batch_size: 16              # 批次大小
  save_dir: "results/finmme"  # 结果保存目录
```

### 6. 特征提取选项

#### 使用预训练模型（推荐，但较慢）

```yaml
data:
  use_pretrained_features: true
```

这会使用 ViT 和 BERT 模型提取特征，性能更好但需要：
- 首次运行会下载预训练模型
- 需要更多内存和计算资源

#### 不使用预训练模型（快速测试）

```yaml
data:
  use_pretrained_features: false
```

这会使用简单的特征提取方法，速度更快但性能较差，适合快速测试。

### 7. 验证数据集

运行前可以验证数据集是否正确加载：

```python
from utils.load_finmme import load_finmme_data

# 加载数据
data = load_finmme_data(
    data_dir="data/finmme",
    splits=["train", "test"],
    use_pretrained_features=False  # 快速测试时设为 False
)

print("训练集样本数:", len(data['train']['labels']))
print("测试集样本数:", len(data['test']['labels']))
print("模态形状:", [mod.shape for mod in data['train']['modalities']])
```

### 8. 常见问题

#### Q: 提示找不到数据文件

**A:** 检查：
1. `data/finmme/train.csv` 或 `data/finmme/train.parquet` 是否存在
2. `data/finmme/test.csv` 或 `data/finmme/test.parquet` 是否存在
3. 配置文件中的 `data_dir` 路径是否正确

#### Q: 图像加载失败

**A:** 检查：
1. CSV/Parquet 文件中的 `image_path` 字段是否正确
2. 图像文件是否存在于 `data/finmme/train/images/` 和 `data/finmme/test/images/` 目录
3. 图像路径是相对路径还是绝对路径

#### Q: 内存不足

**A:** 尝试：
1. 减小 `batch_size`（例如改为 8 或 4）
2. 设置 `use_pretrained_features: false`
3. 只加载部分数据（修改代码）

#### Q: 预训练模型下载失败

**A:** 可以：
1. 设置 `use_pretrained_features: false` 跳过预训练模型
2. 或手动下载模型到本地缓存目录

### 9. 示例命令

```bash
# 进入项目目录
cd quantum_multimodal_comparison

# 训练所有模型（使用 FinMME 数据集）
python train.py --config configs/config_finmme.yaml

# 只训练量子模型（需要修改配置文件中的 models_to_train）
# 或直接修改 config_finmme.yaml，只保留 QuantumHybrid

# 查看结果
# 结果会保存在 results/finmme/ 目录下
```

### 10. 下一步

训练完成后：
- 查看 `results/finmme/` 目录下的训练结果
- 运行 `python compare.py --results_dir results/finmme` 比较模型性能
- 查看生成的图表和报告






