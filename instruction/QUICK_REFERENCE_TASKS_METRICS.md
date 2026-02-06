# FCMR 替代任务与指标快速参考

## 🎯 8个替代任务（快速切换）

| 任务 | 输出维度 | 任务类型 | 主要指标 | 应用场景 |
|------|---------|---------|---------|---------|
| **1. 难度预测** | 3 | 分类 | Accuracy, F1, AUC-ROC | 风险评估、自适应学习 |
| **2. 答案数量预测** | 4 | 分类 | Accuracy, F1, Top-K | 信息检索、问题理解 |
| **3. 模态重要性排序** | 3 | 排序 | NDCG@K, MRR, MAP | 可解释性、资源分配 |
| **4. 置信度预测** | 1 | 回归 | MSE, R², Correlation | 不确定性量化、主动学习 |
| **5. 异常检测** | 1 | 分类 | AUC-ROC, AUC-PR, Precision@K | 欺诈检测、数据质量 |
| **6. 模态对齐质量** | 1 | 回归 | MSE, R², Correlation | 数据质量评估 |
| **7. 答案类型分类** | 3 | 分类 | Accuracy, F1, AUC-ROC | 问题类型识别 |
| **8. 模态缺失鲁棒性** | 原任务 | 分类/回归 | 性能下降率 | 实际部署 |

---

## 📊 扩展指标（不同于原论文）

### 分类任务指标

| 指标 | 范围 | 越高越好 | 说明 |
|------|------|---------|------|
| **AUC-ROC** | [0, 1] | ✅ | ROC曲线下面积，适合不平衡数据 |
| **AUC-PR** | [0, 1] | ✅ | Precision-Recall曲线下面积 |
| **Top-K Accuracy** | [0, 1] | ✅ | Top-K预测准确率（K=2,3,5） |
| **Cohen's Kappa** | [-1, 1] | ✅ | 考虑随机一致性的准确率 |
| **MCC** | [-1, 1] | ✅ | Matthews相关系数，平衡评估 |
| **Hamming Loss** | [0, 1] | ❌ | 多标签分类错误率 |

### 回归任务指标

| 指标 | 范围 | 越高越好 | 说明 |
|------|------|---------|------|
| **Pearson R** | [-1, 1] | ✅ | 线性相关系数 |
| **Spearman R** | [-1, 1] | ✅ | 秩相关系数，单调关系 |
| **Kendall Tau** | [-1, 1] | ✅ | 排序一致性 |
| **MAPE** | [0, ∞) | ❌ | 平均绝对百分比误差 |
| **Information Ratio** | (-∞, ∞) | ✅ | 信息比率（金融） |
| **Hit Rate** | [0, 1] | ✅ | 方向预测准确率 |

### 排序任务指标

| 指标 | 范围 | 越高越好 | 说明 |
|------|------|---------|------|
| **NDCG@K** | [0, 1] | ✅ | 归一化折损累积增益（K=3,5,10） |
| **MRR** | [0, 1] | ✅ | 平均倒数排名 |
| **MAP** | [0, 1] | ✅ | 平均精度均值 |

### 金融专用指标

| 指标 | 范围 | 越高越好 | 说明 |
|------|------|---------|------|
| **Sharpe Ratio** | (-∞, ∞) | ✅ | 夏普比率（收益/风险） |
| **Max Drawdown** | (-∞, 0] | ✅ | 最大回撤（越小越好） |
| **Hit Rate** | [0, 1] | ✅ | 方向预测准确率 |

---

## 🚀 应用场景矩阵

| 应用场景 | 推荐任务 | 推荐指标 | 价值 |
|---------|---------|---------|------|
| **金融风险评估** | 难度预测 + 置信度预测 | Accuracy, AUC-ROC, Information Ratio | 自动化风险分级 |
| **智能问答系统** | 答案数量预测 + 模态重要性 | Accuracy, NDCG@K, MRR | 优化检索策略 |
| **数据质量监控** | 异常检测 + 模态对齐 | AUC-ROC, AUC-PR, Silhouette | 识别噪声数据 |
| **自适应学习** | 难度预测 + 置信度预测 | Accuracy, F1, AUC-ROC | 个性化学习路径 |
| **融合策略优化** | 模态重要性 + 缺失鲁棒性 | NDCG@K, Spearman R | 动态调整融合权重 |
| **趋势预测** | 置信度预测（回归） | Hit Rate, Sharpe Ratio | 交易策略优化 |
| **可解释性分析** | 模态重要性排序 | NDCG@K, 注意力可视化 | 决策解释 |

---

## 💻 快速使用

### 1. 切换任务（在 `load_fcmr.py` 中）

```python
from utils.fcmr_task_switcher import FCMRTaskSwitcher

# 加载数据
df = pd.read_csv('data/fcmr/dataset/easy/easy_data.csv')
switcher = FCMRTaskSwitcher(df)

# 获取难度预测标签
difficulty_labels = switcher.get_labels('difficulty')
task_info = switcher.get_task_info('difficulty')
# task_info: {'output_dim': 3, 'task_type': 'classification', ...}
```

### 2. 计算扩展指标（在 `train.py` 中）

```python
from utils.extended_metrics import calculate_all_extended_metrics

# 计算所有扩展指标
extended_metrics = calculate_all_extended_metrics(
    y_true=val_labels,
    y_pred=val_predictions,
    task_type='classification',  # 或 'regression', 'ranking'
    is_multilabel=False,
    modality_features=[mod1, mod2, mod3],  # 可选：模态重要性分析
    k=5  # Top-K
)

# extended_metrics 包含：
# - AUC-ROC, AUC-PR
# - Top-K Accuracy
# - Cohen's Kappa, MCC
# - NDCG@K, MRR, MAP (如果适用)
# - Pearson/Spearman/Kendall (如果回归)
# - Modality Importance (如果提供模态特征)
```

### 3. 配置文件示例

```yaml
# configs/config_fcmr_difficulty.yaml
task:
  name: difficulty_prediction
  output_dim: 3
  task_type: classification
  
metrics:
  standard: [Accuracy, F1_Macro, F1_Micro]
  extended: [AUC-ROC, AUC-PR, Top-5_Accuracy, Cohen_Kappa, MCC]
```

---

## 📝 论文写作模板

### 实验部分

> **Alternative Task Evaluations**
> 
> We evaluate our models on several alternative tasks derived from FCMR dataset, focusing on practical applications rather than direct comparison with LLM baselines:
> 
> 1. **Difficulty Prediction**: Classifying questions into Easy/Medium/Hard (3-class). Metrics: Accuracy=0.XX, F1_Macro=0.XX, AUC-ROC=0.XX.
> 
> 2. **Answer Count Prediction**: Predicting the number of correct answers (0-3). Metrics: Accuracy=0.XX, Top-3_Accuracy=0.XX.
> 
> 3. **Modality Importance Ranking**: Ranking the importance of text/table/chart modalities. Metrics: NDCG@3=0.XX, MRR=0.XX.
> 
> 4. **Answer Confidence Prediction**: Predicting model confidence as a regression task. Metrics: R²=0.XX, Spearman_R=0.XX, Information_Ratio=0.XX.
> 
> These tasks demonstrate the versatility of our fusion models beyond the original multi-label classification setup.

---

## 🔗 相关文件

- `utils/extended_metrics.py` - 扩展指标实现
- `utils/fcmr_task_switcher.py` - 任务切换器
- `instruction/FCMR_ALTERNATIVE_TASKS.md` - 详细任务说明
- `configs/config_fcmr_*.yaml` - 配置文件模板

---

**最后更新**: 2026-01-29
