# Quantum Hybrid Model for Multimodal Fusion Comparison

本项目使用量子混合模型比较多种多模态融合基线模型，包括：

- **TFN (Tensor Fusion Network)**: 通过张量融合层建模模态间交互
- **LMF (Low-rank Multimodal Fusion)**: 在 TFN 基础上引入低秩分解，降低计算复杂度
- **MFN (Memory Fusion Network)**: 引入记忆机制捕捉跨模态的长期依赖
- **MulT (Multimodal Transformer)**: 基于多头跨模态注意力机制
- **Graph-based baselines**: GCN、Hypergraph NN，用于建模模态间拓扑关系
- **Quantum Hybrid Model**: 使用量子混合模型进行多模态融合

## 项目结构

```
quantum_multimodal_comparison/
├── models/
│   ├── __init__.py
│   ├── tfn.py              # Tensor Fusion Network
│   ├── lmf.py              # Low-rank Multimodal Fusion
│   ├── mfn.py              # Memory Fusion Network
│   ├── mult.py             # Multimodal Transformer
│   ├── graph_baselines.py  # GCN, Hypergraph NN
│   └── quantum_hybrid.py   # Quantum Hybrid Model
├── utils/
│   ├── __init__.py
│   ├── data_loader.py      # 数据加载器
│   └── metrics.py          # 评估指标
├── configs/
│   └── config.yaml         # 配置文件
├── train.py                # 训练脚本
├── compare.py              # 模型比较脚本
└── requirements.txt        # 依赖包

```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 安装依赖

```bash
cd quantum_multimodal_comparison
pip install -r requirements.txt
```

### 2. 测试模型

首先测试所有模型是否能正常工作：

```bash
python test_models.py
```

### 3. 训练所有模型

训练配置文件中指定的所有模型：

```bash
# 完整配置（推荐用于正式实验）
python train.py --config configs/config.yaml

# 快速测试配置（模型更小，训练更快）
python train.py --config configs/config_quick.yaml
```

或者使用一键运行脚本（包含测试、训练和比较）：

```bash
python run_all.py
```

训练过程会：
- 自动生成合成数据（可在配置文件中修改）
- 训练所有指定的模型
- 保存最佳模型权重到 `results/` 目录
- 保存训练历史和指标

### 4. 比较模型性能

训练完成后，比较所有模型的性能：

```bash
python compare.py --results_dir results/
```

这会生成：
- `comparison_table.csv`: 详细的性能对比表
- `comparison_bar.png`: 柱状图对比
- `comparison_radar.png`: 雷达图对比

### 5. 使用 FinMME 数据集

FinMME 是一个金融多模态问答数据集，包含图像、文本和选项三个模态。

#### 5.1 下载数据集

首先登录 HuggingFace 并下载数据集：

```bash
# 登录 HuggingFace
huggingface-cli login

# 下载数据集到本地
python utils/download_finmme.py --output_dir data/finmme --splits train test --verify
```

#### 5.2 训练模型

使用 FinMME 数据集训练：

```bash
python train.py --config configs/config_finmme.yaml
```

详细说明请参考 [FINMME_GUIDE.md](FINMME_GUIDE.md)

### 6. 使用自己的数据

#### 方法1：修改 train.py（推荐）

在 `train.py` 中替换数据生成部分：

```python
# 原来的合成数据生成
# modalities, labels = generate_synthetic_data(...)

# 替换为你的数据加载函数
from utils.load_real_data import load_from_csv, load_from_numpy, load_time_series_data

# 示例1：从CSV加载
modalities, labels = load_from_csv(
    csv_paths=['data/modality1.csv', 'data/modality2.csv', 'data/modality3.csv'],
    label_path='data/labels.csv'
)

# 示例2：从NPZ加载
modalities, labels = load_from_numpy('data/multimodal_data.npz')

# 示例3：时间序列数据
modalities, labels = load_time_series_data('data/', modalities=['text', 'audio', 'video'])
```

#### 方法2：数据格式要求

你的数据需要满足以下格式：

- **modalities**: list of numpy arrays
  - 每个数组形状: `(n_samples, seq_len, feature_dim)` 或 `(n_samples, feature_dim)`
  - 所有模态的 `n_samples` 必须相同
  
- **labels**: numpy array
  - 形状: `(n_samples, output_dim)`
  - 对于回归任务，`output_dim=1`

#### 方法3：创建自定义数据加载器

参考 `utils/load_real_data.py` 中的示例函数，创建适合你数据格式的加载函数。

## 模型说明

### TFN (Tensor Fusion Network)
使用张量外积构建多模态融合表示，捕获所有模态间的交互。

### LMF (Low-rank Multimodal Fusion)
在TFN基础上使用低秩分解降低计算复杂度，同时保持融合能力。

### MFN (Memory Fusion Network)
引入记忆网络机制，捕捉跨模态的长期依赖关系。

### MulT (Multimodal Transformer)
基于Transformer架构，使用多头跨模态注意力机制进行融合。

### Graph-based Models
- **GCN**: 图卷积网络，将模态视为图节点
- **Hypergraph NN**: 超图神经网络，建模高阶模态关系

### Quantum Hybrid Model
使用量子混合模型进行多模态融合，利用量子计算的并行性和纠缠特性。
 
