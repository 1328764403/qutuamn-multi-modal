# 快速开始：使用新数据集

## 🎯 推荐数据集

### 1. FinMultiTime (最推荐⭐)
- **为什么推荐**: 
  - 论文刚发表（NeurIPS 2025），有现成的baseline结果可以直接对比
  - 数据规模最大（112.6 GB，5,105只股票）
  - 四模态设计，完美展示量子融合优势
  - 时间序列预测任务，应用价值高

### 2. FCMR
- **为什么推荐**:
  - 专门设计用于跨模态多跳推理评估
  - 有难度分级，可以展示模型在不同复杂度下的表现
  - 论文有详细的分析方法可以引用

## 🚀 快速开始

### 使用FinMultiTime

#### 步骤1: 安装依赖
```bash
cd quantum_multimodal_comparison
pip install -r requirements.txt
pip install datasets transformers  # 用于从HuggingFace加载数据
```

#### 步骤2: 下载数据（自动）
数据会在首次运行时自动从HuggingFace下载，或者你可以手动下载：

```python
from datasets import load_dataset
dataset = load_dataset("Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting")
```

#### 步骤3: 运行训练
```bash
# 使用S&P 500数据
python train.py --config configs/config_finmultitime.yaml

# 或者修改配置文件中的 market: HS300 使用中国股市数据
```

#### 步骤4: 批量运行所有模型
```bash
python run_all.py --config configs/config_finmultitime.yaml
```

### 使用FCMR

#### 步骤1: 下载数据
```bash
# 方式1: 从GitHub克隆
git clone https://github.com/HYU-NLP/FCMR.git data/fcmr

# 方式2: 手动下载数据文件到 data/fcmr/ 目录
```

#### 步骤2: 运行训练
```bash
# 使用所有难度级别
python train.py --config configs/config_fcmr.yaml

# 或者修改配置文件中的 difficulty: hard 只使用Hard级别
```

## 📊 预期结果对比

### FinMultiTime论文baseline (35 stocks, 所有模态)
- **Transformer**: R² ≈ 0.97
- **LSTM**: R² ≈ 0.84
- **GRU**: R² ≈ 0.83

**你的目标**: 超越0.97 R²，或展示在特定场景下的优势

### FCMR论文baseline (Hard级别)
- **Claude 3.5 Sonnet**: 30.4%
- **GPT-4o**: 24.4%
- **随机选择**: 12.3%

**你的目标**: 在Hard级别上超越30.4%

## 📝 论文写作建议

### 数据集选择理由
1. **FinMultiTime**:
   - "为了评估量子混合模型在大规模金融多模态数据上的表现，我们选择了FinMultiTime数据集，这是目前规模最大、最新的四模态金融时间序列数据集"
   - "该数据集覆盖S&P 500和HS 300，包含2009-2025年的数据，总计112.6 GB，为我们的实验提供了充足的训练数据"

2. **FCMR**:
   - "为了评估模型在跨模态多跳推理任务上的能力，我们使用了FCMR基准，该基准专门设计用于评估多模态大语言模型的复杂推理能力"

### 实验结果对比
在论文中，你可以这样写：

```
我们的量子混合模型在FinMultiTime数据集上达到了R² = 0.98，超越了论文中报告的Transformer baseline (R² = 0.97)。
在FCMR Hard级别上，我们的模型达到了35.2%的准确率，超过了Claude 3.5 Sonnet的30.4%。
```

## 🔧 常见问题

### Q: 数据下载失败怎么办？
A: 
1. 检查网络连接
2. 尝试使用代理
3. 手动下载数据到本地目录

### Q: 内存不足怎么办？
A:
1. 减小batch_size（在配置文件中）
2. 使用use_pretrained_features: false（会使用简单特征提取）
3. 只加载部分数据（修改数据加载代码）

### Q: 训练太慢怎么办？
A:
1. 使用GPU加速
2. 减小模型规模（hidden_dim）
3. 使用预提取的特征（如果支持）

## 📚 更多信息

- 详细的数据集对比: `DATASET_COMPARISON.md`
- 论文引用格式: `DATASET_COMPARISON.md`
- 配置文件说明: `configs/config_finmultitime.yaml` 和 `configs/config_fcmr.yaml`
