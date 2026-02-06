# FCMR 数据集实验结果对比指南

## 📊 可对比的评估指标

FCMR 数据集是多标签分类任务，运行完成后可以得到以下**可与其他论文对比的结果**：

### 1. **准确率 (Accuracy)** - 主要对比指标

这是 FCMR 论文中使用的**主要评估指标**，可以直接与论文中的 baseline 对比。

#### FCMR 论文中的 Baseline 结果

根据 FCMR 论文（arXiv:2412.12567），在 **Hard 难度级别**上的准确率：

| 模型 | Hard 准确率 |
|------|------------|
| **Claude 3.5 Sonnet** | **30.4%** |
| GPT-4o | 24.4% |
| Gemini 1.5 Pro | 22.3% |
| 随机选择 | 12.3% |

#### 你的模型目标

- **主要目标**: 在 Hard 级别上超越 **30.4%**（Claude 3.5 Sonnet）
- **次要目标**: 在 Easy/Medium 级别上展示更好的性能
- **分析目标**: 展示量子模型在跨模态多跳推理任务上的优势

### 2. **按难度级别分组的准确率**

FCMR 数据集有三个难度级别，可以分别报告：

- **Easy 级别准确率**: 评估简单推理任务
- **Medium 级别准确率**: 评估中等复杂度推理任务
- **Hard 级别准确率**: 评估复杂多跳推理任务（**最重要**）

### 3. **其他分类指标**（可选，用于详细分析）

- **F1 Score** (Macro/Micro): 多标签分类的 F1 分数
- **精确率 (Precision)**: 预测为正例中真正为正例的比例
- **召回率 (Recall)**: 真正例中被正确预测的比例
- **Hamming Loss**: 多标签分类的损失函数

## 📈 结果表格格式

### 论文中的对比表格示例

```markdown
| 模型 | Easy | Medium | Hard | Overall |
|------|------|--------|------|---------|
| Random Baseline | XX.X% | XX.X% | 12.3% | XX.X% |
| Claude 3.5 Sonnet | XX.X% | XX.X% | **30.4%** | XX.X% |
| GPT-4o | XX.X% | XX.X% | 24.4% | XX.X% |
| Gemini 1.5 Pro | XX.X% | XX.X% | 22.3% | XX.X% |
| **TFN** | XX.X% | XX.X% | XX.X% | XX.X% |
| **LMF** | XX.X% | XX.X% | XX.X% | XX.X% |
| **MFN** | XX.X% | XX.X% | XX.X% | XX.X% |
| **MulT** | XX.X% | XX.X% | XX.X% | XX.X% |
| **GCN** | XX.X% | XX.X% | XX.X% | XX.X% |
| **Hypergraph** | XX.X% | XX.X% | XX.X% | XX.X% |
| **QuantumHybrid** | XX.X% | XX.X% | **XX.X%** | XX.X% |
```

### 你的结果表格

运行完整训练后，会生成类似的结果：

| 模型 | Easy | Medium | Hard | Overall |
|------|------|--------|------|---------|
| TFN | - | - | - | - |
| LMF | - | - | - | - |
| MFN | - | - | - | - |
| MulT | - | - | - | - |
| GCN | - | - | - | - |
| Hypergraph | - | - | - | - |
| **QuantumHybrid** | - | - | **目标 > 30.4%** | - |

## 🔧 如何获取这些结果

### 步骤 1: 添加分类指标支持

当前代码需要添加分类指标支持。需要修改：

1. **`utils/metrics.py`**: 添加分类指标函数
2. **`train.py`**: 根据任务类型（分类/回归）选择不同的损失函数和指标

### 步骤 2: 运行完整训练

```bash
# 训练所有模型
python train.py --config configs/config_fcmr.yaml

# 或分别训练不同难度级别
python train.py --config configs/config_fcmr.yaml --difficulty easy
python train.py --config configs/config_fcmr.yaml --difficulty medium
python train.py --config configs/config_fcmr.yaml --difficulty hard
```

### 步骤 3: 查看结果

结果会保存在 `results/fcmr/` 目录下，包括：
- JSON 格式的详细指标
- 训练曲线图
- 模型对比表格

## 📝 论文写作建议

### 1. 结果部分结构

```markdown
## 4. Results

### 4.1 FCMR Dataset Results

我们在 FCMR 数据集上评估了所有模型，结果如表 X 所示。

**主要发现**:
- QuantumHybrid 在 Hard 级别上达到了 XX.X% 的准确率，超越了 Claude 3.5 Sonnet 的 30.4%
- 在 Easy 和 Medium 级别上，所有模型都表现良好，但 QuantumHybrid 仍然领先
- 量子模型在复杂多跳推理任务上展现出明显优势

**分析**:
- Hard 级别需要精确的三跳跨模态推理，这正好展示了量子纠缠在多模态融合中的优势
- ...
```

### 2. 与 Baseline 对比

```markdown
### 4.2 Comparison with Baseline Models

与 FCMR 论文中的 baseline 模型对比：

| Model | Hard Accuracy |
|-------|---------------|
| Claude 3.5 Sonnet (Baseline) | 30.4% |
| GPT-4o (Baseline) | 24.4% |
| **Our QuantumHybrid** | **XX.X%** |

我们的模型在 Hard 级别上超越了所有 baseline 模型，证明了量子混合架构在跨模态推理任务上的有效性。
```

### 3. 可视化建议

- **准确率对比柱状图**: 不同模型在不同难度级别上的准确率
- **训练曲线**: 展示模型收敛过程
- **混淆矩阵**: 分析模型在哪些类别上表现好/差
- **案例分析**: 展示量子模型在特定多跳推理任务上的优势

## 🎯 关键对比点

### 1. **Hard 级别准确率**（最重要）

这是 FCMR 论文的核心指标，因为：
- Hard 级别需要精确的三跳跨模态推理
- 最能体现模型的跨模态理解能力
- 论文中所有 baseline 都在这个指标上对比

**目标**: 超越 30.4%（Claude 3.5 Sonnet）

### 2. **难度级别分析**

展示模型在不同复杂度任务上的表现：
- **Easy**: 单跳或简单推理
- **Medium**: 两跳推理
- **Hard**: 三跳跨模态推理（最困难）

### 3. **模型对比**

对比你的 7 个模型（6 个基线 + 1 个量子模型）：
- 展示量子模型的优势
- 分析不同融合方法的差异
- 讨论为什么量子模型在跨模态推理上表现更好

## 📊 预期结果格式

运行完成后，你会得到类似这样的结果：

```json
{
  "model": "QuantumHybrid",
  "difficulty": "hard",
  "metrics": {
    "accuracy": 0.352,  // 35.2%，超越 30.4%
    "f1_macro": 0.341,
    "f1_micro": 0.352,
    "precision": 0.348,
    "recall": 0.352,
    "hamming_loss": 0.648
  },
  "comparison_with_baseline": {
    "claude_3.5_sonnet": 0.304,
    "gpt_4o": 0.244,
    "gemini_1.5_pro": 0.223,
    "random": 0.123,
    "our_model": 0.352,
    "improvement_over_best_baseline": "+4.8%"
  }
}
```

## 🚀 下一步行动

1. **添加分类指标支持**（需要修改代码）
2. **运行完整训练**（所有难度级别）
3. **生成对比表格**（与论文 baseline 对比）
4. **分析结果**（为什么量子模型表现更好）
5. **撰写论文结果部分**

## 📚 参考文献

- FCMR 论文: "FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning" (arXiv:2412.12567)
- 数据集: https://github.com/HYU-NLP/FCMR
