# 代码中实际使用的数据集说明

## 📊 数据集概览

代码支持**4种数据集**，用于比较不同多模态融合模型的性能：

### 1. 合成数据 (Synthetic Data) - 默认配置

**用途**: 快速测试和验证模型架构

**数据内容**:
- **模态数量**: 3个模态（可配置）
- **模态类型**: 随机生成的多维特征序列
- **数据格式**: 
  - 每个模态: `(n_samples, seq_len, feature_dim)`
  - 默认: 1000个样本，序列长度[10,15,12]，特征维度[32,64,48]
- **标签**: 基于模态数据的线性组合 + 10%噪声
- **任务类型**: 回归任务（预测连续值）

**特点**:
- ✅ 快速生成，无需下载
- ✅ 用于验证模型是否能学习多模态关系
- ⚠️ 数据简单，不适合真实场景评估

**配置文件**: `configs/config.yaml`

---

### 2. FinMME 数据集 ⭐

**用途**: 金融多模态问答任务

**数据内容**:
- **模态数量**: 3个模态
  1. **图像模态**: 金融图表（K线图、技术分析图等）
  2. **文本模态**: 问题描述文本
  3. **选项模态**: 4个选项文本（A/B/C/D）
- **数据规模**: 
  - 训练集: ~8,000+ 样本
  - 测试集: ~3,000+ 样本
  - 18个金融领域，6个资产类别
- **任务类型**: 分类任务（4选1）
- **标签**: 0-3（对应选项索引）

**数据来源**:
- HuggingFace: `luojunyu/FinMME`
- 论文: "FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation" (ACL 2025)

**特点**:
- ✅ 真实金融数据
- ✅ 多模态对齐
- ✅ 适合金融问答场景
- ⚠️ 规模相对较小

**配置文件**: `configs/config_finmme.yaml`

**数据示例**:
```
图像: [金融K线图]
问题: "Which company has the highest revenue in 2023?"
选项: ["Company A", "Company B", "Company C", "Company D"]
答案: 2 (Company C)
```

---

### 3. FinMultiTime 数据集 ⭐⭐⭐

**用途**: 金融时间序列预测（四模态）

**数据内容**:
- **模态数量**: 4个模态
  1. **文本模态**: 金融新闻文本
  2. **表格模态**: 结构化财务表格数据
  3. **图像模态**: K线技术图表
  4. **时间序列模态**: 股价时间序列数据
- **数据规模**: 
  - **S&P 500**: 5,105只股票，2009-2025年
  - **HS 300**: 中国A股，2009-2025年
  - 112.6 GB数据，分钟级/日级/季度级分辨率
- **任务类型**: 回归任务（股价预测）
- **标签**: 连续值（股价或收益率）

**数据来源**:
- HuggingFace: `Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting`
- 论文: "FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis" (NeurIPS 2025)

**特点**:
- ✅ **规模最大**（数百万条记录）
- ✅ **四模态对齐**，适合多模态融合研究
- ✅ **双语数据**（中英文）
- ✅ **时间序列预测**，更贴近实际应用
- ✅ 论文刚发表，有现成baseline对比

**配置文件**: `configs/config_finmultitime.yaml`

**数据示例**:
```
文本: "Apple Inc. reported strong Q4 earnings..."
表格: [财务指标表格数据]
图像: [K线图]
时间序列: [过去30天股价序列]
标签: 150.25 (预测的收盘价)
```

---

### 4. FCMR 数据集 ⭐⭐

**用途**: 金融跨模态多跳推理

**数据内容**:
- **模态数量**: 3个模态
  1. **文本模态**: 金融报告文本
  2. **表格模态**: 财务数据表格
  3. **图表模态**: 金融图表
- **数据规模**: 
  - Easy: 757样本
  - Medium: 728样本
  - Hard: 714样本
  - 总计: 2,199样本
- **任务类型**: 多标签分类（0-3个正确答案）
- **难度级别**: Easy / Medium / Hard
  - **Hard级别**: 需要精确的三跳跨模态推理

**数据来源**:
- GitHub: https://github.com/HYU-NLP/FCMR
- 论文: "FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning" (arXiv 2024)

**特点**:
- ✅ **专门设计用于跨模态推理评估**
- ✅ **难度分级**，可以评估模型在不同复杂度下的表现
- ✅ **避免数据污染**（使用真实金融数据）
- ✅ 论文有详细分析，可引用其方法
- ⚠️ 样本数量相对较少

**配置文件**: `configs/config_fcmr.yaml`

**数据示例**:
```
文本: "Company X reported revenue increase..."
表格: [财务数据表格]
图表: [趋势图]
问题: "Which companies meet the criteria?"
答案: [0, 1, 2] (多标签，3个公司都符合)
```

---

## 🔄 数据集切换

### 默认使用合成数据
```bash
python train.py --config configs/config.yaml
```

### 切换到FinMME
```bash
python train.py --config configs/config_finmme.yaml
```

### 切换到FinMultiTime
```bash
python train.py --config configs/config_finmultitime.yaml
```

### 切换到FCMR
```bash
python train.py --config configs/config_fcmr.yaml
```

---

## 📈 数据集对比总结

| 数据集 | 模态数 | 任务类型 | 数据规模 | 推荐场景 |
|--------|--------|----------|----------|----------|
| **合成数据** | 3 | 回归 | 1,000样本 | 快速测试、验证架构 |
| **FinMME** | 3 | 分类 | ~11,000样本 | 金融问答、多模态理解 |
| **FinMultiTime** | 4 | 回归 | 数百万条 | **时间序列预测、大规模实验** ⭐ |
| **FCMR** | 3 | 多标签分类 | 2,199样本 | **跨模态推理评估** ⭐ |

---

## 🎯 推荐使用场景

### 场景1: 快速验证模型架构
**推荐**: 合成数据
- 无需下载，快速运行
- 验证模型是否能学习多模态关系

### 场景2: 金融问答任务
**推荐**: FinMME
- 真实金融数据
- 适合展示金融应用价值

### 场景3: 大规模时间序列预测（推荐）⭐⭐⭐
**推荐**: FinMultiTime
- 数据规模最大
- 四模态设计，可以展示量子融合优势
- 有现成baseline对比（Transformer R²≈0.97）
- 时间序列预测任务，应用价值高

### 场景4: 跨模态推理能力评估
**推荐**: FCMR
- 专门设计用于多跳推理评估
- 有难度分级，可以展示模型在不同复杂度下的表现

---

## 📝 数据格式说明

### 所有数据集统一格式

**输入格式**:
```python
modalities = [
    mod1,  # shape: (n_samples, seq_len1, feature_dim1)
    mod2,  # shape: (n_samples, seq_len2, feature_dim2)
    mod3,  # shape: (n_samples, seq_len3, feature_dim3)
    ...
]

labels = np.array(...)  # shape: (n_samples, output_dim)
```

**特征提取**:
- 使用预训练模型（BERT/ViT）提取特征
- 特征维度: 768（默认）
- 可以禁用特征提取，直接使用原始数据

---

## 🔍 实际使用的数据

根据你的代码配置，**默认使用合成数据**进行模型对比。如果要使用真实数据集，需要：

1. **下载数据集**（FinMME/FinMultiTime/FCMR）
2. **修改配置文件**中的`data.source`字段
3. **运行训练脚本**

**当前默认配置** (`configs/config.yaml`):
- 数据源: `synthetic`（合成数据）
- 样本数: 1000
- 模态数: 3
- 任务: 回归

---

## 📚 相关文档

- `DATASET_COMPARISON.md` - 数据集详细对比
- `BASELINE_PAPERS.md` - 数据集论文引用
- `LOCAL_DATA_GUIDE.md` - 本地数据使用指南
- `FINMME_GUIDE.md` - FinMME数据集使用指南
- `SETUP_FINMULTITIME.md` - FinMultiTime数据集设置指南

---

**最后更新**: 2026-01-26
