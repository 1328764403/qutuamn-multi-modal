# FCMR 数据集替代任务定义与应用场景

本文档列出基于 FCMR 数据集可以定义的各种任务、指标和应用场景，**不直接对比原论文的 LLM baseline**。

---

## 📋 目录

1. [任务定义](#任务定义)
2. [扩展指标](#扩展指标)
3. [应用场景](#应用场景)
4. [实现建议](#实现建议)

---

## 🎯 任务定义

### 1. 难度预测任务 (Difficulty Prediction)

**任务描述**: 预测题目的难度级别（Easy/Medium/Hard）

**标签构建**:
```python
# 从 FCMR 数据中的 difficulty 字段
difficulty_map = {'easy': 0, 'medium': 1, 'hard': 2}
labels = df['difficulty'].map(difficulty_map).values
```

**输出维度**: `output_dim=3` (三分类)

**适用指标**:
- Accuracy, F1_Macro, F1_Micro
- AUC-ROC (one-vs-rest), AUC-PR
- Top-K Accuracy (K=2, 3)
- Cohen's Kappa, MCC

**应用价值**: 
- 评估模型对题目复杂度的理解能力
- 可用于自适应学习系统
- 金融风险评估（简单→高风险，复杂→需深入分析）

---

### 2. 答案数量预测 (Answer Count Prediction)

**任务描述**: 预测正确答案的数量（0-3个）

**标签构建**:
```python
# 从 answer 字段解析答案数量
def count_answers(answer_str):
    if answer_str == 'None' or pd.isna(answer_str):
        return 0
    return len(str(answer_str).split(','))
labels = df['answer'].apply(count_answers).values
```

**输出维度**: `output_dim=4` (0, 1, 2, 3个答案)

**适用指标**:
- Accuracy, F1_Macro
- AUC-ROC, AUC-PR
- Mean Absolute Error (作为回归任务)
- Spearman Correlation (如果作为排序任务)

**应用价值**:
- 评估模型对问题复杂度的量化理解
- 信息检索：预测需要检索多少个相关文档
- 金融分析：预测需要关注多少个关键指标

---

### 3. 模态重要性排序 (Modality Importance Ranking)

**任务描述**: 预测哪个模态（文本/表格/图表）对答案最重要

**标签构建**:
```python
# 基于答案与各模态的相关性构建标签
# 方法1: 使用注意力权重（需要训练后分析）
# 方法2: 基于答案类型（文本答案→文本模态重要，数值答案→表格重要）
modality_importance = [0.4, 0.3, 0.3]  # [text, table, chart]
```

**输出维度**: `output_dim=3` (每个模态的重要性分数，归一化到[0,1])

**适用指标**:
- NDCG@K (K=3)
- Mean Reciprocal Rank (MRR)
- Mean Average Precision (MAP)
- Spearman/Kendall Correlation (与真实重要性排序)

**应用价值**:
- 可解释性分析：哪些模态对决策最重要
- 资源分配：优先处理重要模态
- 多模态融合策略优化

---

### 4. 答案置信度预测 (Answer Confidence Prediction)

**任务描述**: 预测模型对答案的置信度（回归任务）

**标签构建**:
```python
# 基于难度和答案数量构建置信度标签
# 简单 + 单答案 → 高置信度
# 困难 + 多答案 → 低置信度
def calculate_confidence(difficulty, answer_count):
    base_conf = {'easy': 0.9, 'medium': 0.6, 'hard': 0.3}[difficulty]
    penalty = answer_count * 0.1  # 多答案降低置信度
    return max(0.1, base_conf - penalty)
labels = df.apply(lambda r: calculate_confidence(r['difficulty'], count_answers(r['answer'])), axis=1)
```

**输出维度**: `output_dim=1` (连续值 [0, 1])

**适用指标**:
- MSE, MAE, RMSE, R²
- Pearson/Spearman/Kendall Correlation
- AUC-ROC (转换为二分类：高/低置信度)
- Information Ratio (金融指标)

**应用价值**:
- 不确定性量化：模型何时不确定
- 主动学习：优先标注低置信度样本
- 风险控制：低置信度时触发人工审核

---

### 5. 异常检测任务 (Anomaly Detection)

**任务描述**: 检测多模态数据中的异常样本（与正常模式不符）

**标签构建**:
```python
# 方法1: 基于答案分布（罕见答案组合 → 异常）
answer_counts = df['answer'].value_counts()
rare_threshold = answer_counts.quantile(0.1)
labels = (df['answer'].map(answer_counts) < rare_threshold).astype(int)

# 方法2: 基于模态特征异常（使用 Isolation Forest）
from sklearn.ensemble import IsolationForest
# 提取模态特征后检测异常
```

**输出维度**: `output_dim=1` (二分类：正常/异常) 或 `output_dim=1` (异常分数)

**适用指标**:
- AUC-ROC, AUC-PR
- Precision@K (Top-K 异常检测)
- F1_Score (平衡精确率和召回率)
- Silhouette Score (聚类质量，如果使用无监督)

**应用价值**:
- 金融欺诈检测：异常交易模式
- 数据质量监控：识别错误标注或噪声数据
- 风险预警：异常市场信号

---

### 6. 模态对齐质量评估 (Modality Alignment Quality)

**任务描述**: 评估三个模态之间的对齐/一致性程度

**标签构建**:
```python
# 基于模态特征相似度
def calculate_alignment(text_feat, table_feat, chart_feat):
    # 计算模态间余弦相似度
    sim_text_table = cosine_similarity(text_feat, table_feat)
    sim_text_chart = cosine_similarity(text_feat, chart_feat)
    sim_table_chart = cosine_similarity(table_feat, chart_feat)
    return (sim_text_table + sim_text_chart + sim_table_chart) / 3
```

**输出维度**: `output_dim=1` (对齐分数 [0, 1])

**适用指标**:
- MSE, MAE, R²
- Pearson/Spearman Correlation
- AUC-ROC (转换为二分类：对齐/不对齐)

**应用价值**:
- 数据质量评估：多模态数据是否一致
- 融合策略选择：对齐好的数据用简单融合，对齐差的用复杂融合
- 预训练数据筛选：选择对齐质量高的数据

---

### 7. 答案类型分类 (Answer Type Classification)

**任务描述**: 预测答案类型（单选项/多选项/无答案）

**标签构建**:
```python
def get_answer_type(answer_str):
    if pd.isna(answer_str) or answer_str == 'None':
        return 0  # 无答案
    answer_list = str(answer_str).split(',')
    if len(answer_list) == 1:
        return 1  # 单选项
    else:
        return 2  # 多选项
labels = df['answer'].apply(get_answer_type).values
```

**输出维度**: `output_dim=3` (三分类)

**适用指标**:
- Accuracy, F1_Macro, F1_Micro
- AUC-ROC, AUC-PR
- Top-K Accuracy
- Cohen's Kappa, MCC

**应用价值**:
- 问题类型识别：简单问题 vs 复杂问题
- 检索策略：单答案用精确匹配，多答案用模糊匹配
- 用户界面：根据答案类型调整展示方式

---

### 8. 模态缺失鲁棒性评估 (Missing Modality Robustness)

**任务描述**: 评估模型在某个模态缺失时的性能

**标签构建**:
```python
# 训练时随机 mask 某个模态（置零）
# 测试时评估不同缺失模式下的性能
# 不需要额外标签，使用原始答案标签
```

**输出维度**: 与原始任务相同（分类/回归）

**适用指标**:
- 对比完整模态 vs 缺失模态的性能下降
- 各模型的鲁棒性排名
- 模态重要性分析（缺失哪个模态影响最大）

**应用价值**:
- 实际部署：处理模态缺失场景
- 模型选择：选择对缺失最鲁棒的模型
- 资源优化：优先保证重要模态的质量

---

## 📊 扩展指标

### 分类任务扩展指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| **AUC-ROC** | ROC曲线下面积 | 二分类/多分类（one-vs-rest） |
| **AUC-PR** | Precision-Recall曲线下面积 | 不平衡数据集 |
| **Top-K Accuracy** | Top-K预测准确率 | 允许K个候选答案 |
| **Cohen's Kappa** | 考虑随机一致性的准确率 | 类别不平衡 |
| **MCC** | Matthews相关系数 | 二分类/多分类平衡评估 |
| **Hamming Loss** | 多标签分类错误率 | 多标签任务 |

### 回归任务扩展指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| **Pearson R** | 线性相关系数 | 线性关系评估 |
| **Spearman R** | 秩相关系数 | 单调关系评估 |
| **Kendall Tau** | 排序一致性 | 排序任务 |
| **MAPE** | 平均绝对百分比误差 | 相对误差评估 |
| **Information Ratio** | 信息比率 | 金融预测 |
| **Sharpe Ratio** | 夏普比率 | 金融收益评估 |
| **Hit Rate** | 方向预测准确率 | 趋势预测 |

### 排序任务指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| **NDCG@K** | 归一化折损累积增益 | 推荐系统、检索 |
| **MRR** | 平均倒数排名 | 检索任务 |
| **MAP** | 平均精度均值 | 多标签排序 |

### 聚类/异常检测指标

| 指标 | 说明 | 适用场景 |
|------|------|----------|
| **Silhouette Score** | 轮廓系数 | 聚类质量 |
| **Calinski-Harabasz** | CH指数 | 聚类分离度 |
| **Davies-Bouldin** | DB指数 | 聚类紧密度 |

---

## 🚀 应用场景

### 1. 金融风险评估

**任务**: 难度预测 + 置信度预测

**应用**:
- **风险分级**: Easy → 低风险，Hard → 高风险
- **置信度阈值**: 低置信度时触发人工审核
- **自动化决策**: 高置信度 + Easy → 自动通过

**指标**: Accuracy, AUC-ROC, Information Ratio

---

### 2. 智能问答系统

**任务**: 答案数量预测 + 模态重要性排序

**应用**:
- **检索策略**: 预测需要检索多少个文档
- **模态优先级**: 优先处理重要模态
- **答案生成**: 根据答案数量调整生成策略

**指标**: Accuracy, NDCG@K, MRR

---

### 3. 数据质量监控

**任务**: 异常检测 + 模态对齐质量评估

**应用**:
- **异常样本检测**: 识别标注错误或噪声数据
- **数据一致性检查**: 多模态数据是否对齐
- **预训练数据筛选**: 选择高质量数据

**指标**: AUC-ROC, AUC-PR, Silhouette Score

---

### 4. 自适应学习系统

**任务**: 难度预测 + 答案置信度预测

**应用**:
- **个性化学习路径**: 根据难度调整学习内容
- **主动学习**: 优先标注低置信度样本
- **知识追踪**: 跟踪学习者对不同难度题目的掌握程度

**指标**: Accuracy, F1_Macro, AUC-ROC

---

### 5. 多模态融合策略优化

**任务**: 模态重要性排序 + 模态缺失鲁棒性评估

**应用**:
- **融合权重调整**: 根据重要性动态调整融合权重
- **模型选择**: 选择对缺失最鲁棒的模型
- **资源分配**: 优先保证重要模态的质量

**指标**: NDCG@K, Spearman R, 性能下降率

---

### 6. 金融趋势预测

**任务**: 答案置信度预测（作为回归）+ 方向预测

**应用**:
- **股价趋势**: 预测涨跌方向（Hit Rate）
- **风险评估**: 置信度低时降低仓位
- **交易策略**: 高置信度 + 明确方向 → 执行交易

**指标**: Hit Rate, Information Ratio, Sharpe Ratio

---

### 7. 可解释性分析

**任务**: 模态重要性排序 + 答案类型分类

**应用**:
- **决策解释**: 展示哪些模态对决策最重要
- **用户理解**: 帮助用户理解模型决策过程
- **模型调试**: 识别模型依赖的模态特征

**指标**: NDCG@K, 注意力权重可视化

---

## 💻 实现建议

### 1. 快速切换任务

在 `utils/load_fcmr.py` 中添加任务选择参数：

```python
def load_as_multimodal(self, extract_features=True, task='original'):
    """
    task options:
    - 'original': 原始多标签分类
    - 'difficulty': 难度预测
    - 'answer_count': 答案数量预测
    - 'confidence': 置信度预测
    - 'anomaly': 异常检测
    - 'alignment': 模态对齐质量
    """
    if task == 'difficulty':
        labels = self._encode_difficulty()
    elif task == 'answer_count':
        labels = self._encode_answer_count()
    # ... 其他任务
```

### 2. 指标计算集成

在 `train.py` 中集成扩展指标：

```python
from utils.extended_metrics import calculate_all_extended_metrics

# 计算扩展指标
extended_metrics = calculate_all_extended_metrics(
    y_true=val_labels,
    y_pred=val_predictions,
    task_type=config['task_type'],
    is_multilabel=is_multilabel,
    modality_features=[mod1, mod2, mod3],
    k=5
)
```

### 3. 配置文件扩展

在 `configs/config_fcmr_*.yaml` 中添加任务配置：

```yaml
task:
  name: difficulty_prediction  # 或 answer_count, confidence, etc.
  output_dim: 3
  task_type: classification
  
metrics:
  standard: [Accuracy, F1_Macro, F1_Micro]
  extended: [AUC-ROC, AUC-PR, Top-5_Accuracy, Cohen_Kappa]
  ranking: [NDCG@3, MRR, MAP]  # 如果适用
  correlation: [Pearson_R, Spearman_R]  # 如果适用
```

---

## 📝 论文写作建议

### 实验部分可以这样写：

> **Alternative Task Evaluations on FCMR Dataset**
> 
> Beyond the original multi-label classification task, we evaluate our models on several alternative tasks that better align with practical applications:
> 
> 1. **Difficulty Prediction**: Predicting the difficulty level (Easy/Medium/Hard) of each question, which is useful for adaptive learning systems and risk assessment.
> 
> 2. **Answer Count Prediction**: Predicting how many correct answers exist (0-3), which helps in information retrieval and question understanding.
> 
> 3. **Modality Importance Ranking**: Ranking the importance of each modality (text/table/chart) for the final answer, providing interpretability insights.
> 
> 4. **Answer Confidence Prediction**: Predicting model confidence as a regression task, enabling uncertainty quantification and active learning.
> 
> We report **AUC-ROC**, **NDCG@K**, **Spearman correlation**, and **Top-K accuracy** metrics, which are more suitable for these tasks than the original FCMR metrics.

---

**最后更新**: 2026-01-29
