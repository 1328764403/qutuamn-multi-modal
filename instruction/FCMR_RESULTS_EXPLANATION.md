# FCMR 数据集实验结果说明

## 📊 可以获得的对比结果

运行完 FCMR 数据集训练后，你可以得到以下可以与论文对比的结果：

### 1. **准确率（Accuracy）** - 主要对比指标

FCMR 论文中报告的主要指标是**准确率**，特别是 Hard 难度级别的准确率。

#### FCMR 论文中的 Baseline 结果：

| 模型 | Easy | Medium | Hard |
|------|------|--------|------|
| **Claude 3.5 Sonnet** | - | - | **30.4%** |
| **GPT-4o** | - | - | **24.4%** |
| **Gemini 1.5 Pro** | - | - | **22.3%** |
| **随机选择** | - | - | **12.3%** |

#### 你的模型需要对比的指标：

1. **整体准确率**（所有难度级别）
2. **Easy 难度准确率**
3. **Medium 难度准确率**
4. **Hard 难度准确率** ⭐ **最重要**

**目标**：在 Hard 难度上超越 30.4%（Claude 3.5 Sonnet），或展示在特定推理任务上的优势。

### 2. **其他分类指标**

除了准确率，还可以报告：

- **F1 Score**（宏平均和微平均）
- **精确率（Precision）**
- **召回率（Recall）**
- **每个类别的性能**（8个答案类别：None, 1, 2, 3, 1,2, 1,3, 2,3, 1,2,3）

### 3. **模型对比结果**

你可以对比以下模型：

| 模型 | Easy | Medium | Hard | 整体 |
|------|------|--------|------|------|
| TFN | ? | ? | ? | ? |
| LMF | ? | ? | ? | ? |
| MFN | ? | ? | ? | ? |
| MulT | ? | ? | ? | ? |
| GCN | ? | ? | ? | ? |
| Hypergraph | ? | ? | ? | ? |
| **QuantumHybrid** ⭐ | ? | ? | ? | ? |

### 4. **难度级别分析**

FCMR 数据集的核心价值在于**难度分级**，可以展示：

- **Easy**: 单跳推理（只使用一个模态）
- **Medium**: 两跳推理（使用两个模态）
- **Hard**: 三跳推理（需要跨三个模态的复杂推理）

你的模型在不同难度级别上的表现可以说明：
- 模型处理简单任务的能力
- 模型处理复杂跨模态推理的能力
- 量子模型在复杂推理任务上的优势

## 📈 如何生成对比结果

### 步骤 1: 修改评估指标

当前代码使用的是回归指标（MSE, R2），需要添加分类指标。需要修改：

1. **`utils/metrics.py`** - 添加分类指标函数
2. **`train.py`** - 根据任务类型（分类/回归）选择不同的损失函数和评估指标

### 步骤 2: 运行训练

```bash
# 训练所有模型
python train.py --config configs/config_fcmr.yaml

# 或者只训练特定难度级别
# 修改 config_fcmr.yaml 中的 difficulty: easy/medium/hard
```

### 步骤 3: 生成对比表格

训练完成后，结果会保存在 `results/fcmr/` 目录下，包含：
- 每个模型的准确率
- 每个难度级别的准确率
- 训练曲线
- 模型参数数量

### 步骤 4: 与论文对比

创建一个对比表格，格式如下：

```
Table: Performance Comparison on FCMR Dataset

| Model | Easy | Medium | Hard | Overall |
|-------|------|--------|------|---------|
| Claude 3.5 Sonnet* | - | - | 30.4% | - |
| GPT-4o* | - | - | 24.4% | - |
| Gemini 1.5 Pro* | - | - | 22.3% | - |
| Random Baseline* | - | - | 12.3% | - |
| TFN | X% | X% | X% | X% |
| LMF | X% | X% | X% | X% |
| MFN | X% | X% | X% | X% |
| MulT | X% | X% | X% | X% |
| GCN | X% | X% | X% | X% |
| Hypergraph | X% | X% | X% | X% |
| QuantumHybrid | X% | X% | X% | X% |

* Results from FCMR paper (Kim et al., 2024)
```

## 🎯 论文中可以强调的点

### 1. **量子模型在复杂推理任务上的优势**

如果 QuantumHybrid 在 Hard 难度上表现更好，可以强调：
- 量子叠加态能够同时考虑多个模态的多种可能性
- 量子纠缠能够更好地建模跨模态关系
- 在需要多跳推理的复杂任务上，量子模型有天然优势

### 2. **难度分级分析**

展示模型在不同难度级别上的表现：
- Easy: 所有模型表现都很好（接近 100%）
- Medium: 开始出现差异
- Hard: 量子模型可能显示出优势

### 3. **与 SOTA 大模型对比**

虽然你的模型是轻量级的神经网络模型，而 FCMR 论文测试的是大语言模型（LLM），但可以：
- 强调模型效率（参数量、推理速度）
- 强调在特定任务上的专业化优势
- 展示量子模型在资源受限场景下的潜力

## 📝 需要添加的代码

### 1. 分类指标函数

在 `utils/metrics.py` 中添加：

```python
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report

def calculate_classification_metrics(y_true, y_pred):
    """
    计算多标签分类指标
    
    Args:
        y_true: 真实标签 (n_samples, n_classes) - one-hot 编码
        y_pred: 预测标签 (n_samples, n_classes) - 概率或 one-hot 编码
    """
    # 转换为类别索引
    y_true_classes = np.argmax(y_true, axis=1)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    accuracy = accuracy_score(y_true_classes, y_pred_classes)
    f1_macro = f1_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    f1_micro = f1_score(y_true_classes, y_pred_classes, average='micro', zero_division=0)
    precision = precision_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    recall = recall_score(y_true_classes, y_pred_classes, average='macro', zero_division=0)
    
    return {
        'Accuracy': float(accuracy),
        'F1_Macro': float(f1_macro),
        'F1_Micro': float(f1_micro),
        'Precision': float(precision),
        'Recall': float(recall)
    }
```

### 2. 修改训练脚本

在 `train.py` 中，根据 `output_dim` 判断任务类型：
- 如果 `output_dim > 1` 且是分类任务，使用 `CrossEntropyLoss` 或 `BCELoss`
- 如果 `output_dim == 1`，使用 `MSELoss`（回归任务）

## 🚀 快速开始

1. **添加分类指标支持**（需要修改代码）
2. **运行训练**：
   ```bash
   python train.py --config configs/config_fcmr.yaml
   ```
3. **分析结果**：查看 `results/fcmr/` 目录下的结果文件
4. **生成对比表格**：使用 `generate_paper_tables.py` 生成论文表格

## 📌 注意事项

1. **任务类型**：FCMR 是多标签分类任务（8个类别），不是回归任务
2. **损失函数**：应该使用 `CrossEntropyLoss` 或 `BCELoss`，而不是 `MSELoss`
3. **评估指标**：主要使用准确率，而不是 R2 或 RMSE
4. **难度级别**：需要分别报告每个难度级别的结果

## 📚 参考文献

- FCMR 论文: "FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning" (Kim et al., 2024)
- 数据集: https://github.com/HYU-NLP/FCMR
