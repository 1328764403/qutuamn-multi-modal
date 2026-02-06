# 金融多模态数据集对比与论文引用指南

## 📊 可用数据集对比

### 1. FinMME (当前使用)
- **论文**: "FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation"
- **会议**: ACL 2025
- **数据集**: https://huggingface.co/datasets/luojunyu/FinMME
- **特点**:
  - 金融多模态问答数据集
  - 包含：金融图表图像、问题文本、选项文本
  - 任务：金融多模态问答（4选1）
  - 数据规模：11,000+ 样本，18个金融领域，6个资产类别
- **模态**: 图像、文本、选项（3模态）
- **任务类型**: 分类（多选一）
- **优势**: 已集成，可直接使用
- **劣势**: 规模相对较小，主要是问答任务

### 2. FinMultiTime (推荐⭐)
- **论文**: "FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis"
- **会议**: NeurIPS 2025 (投稿中)
- **数据集**: https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting
- **特点**:
  - **首个大规模四模态金融时间序列数据集**
  - 覆盖S&P 500和HS 300，2009-2025年
  - 112.6 GB数据，分钟级/日级/季度级分辨率
  - 数据规模：5,105只股票，数百万条记录
- **模态**: 
  1. 金融新闻文本（Text）
  2. 结构化财务表格（Table）
  3. K线技术图表（Image）
  4. 股价时间序列（Time Series）
- **任务类型**: 回归（股价预测）、分类（趋势预测）
- **优势**: 
  - ✅ 规模最大，数据最新（2025年）
  - ✅ 四模态对齐，适合多模态融合研究
  - ✅ 双语（中英文），覆盖两个主要市场
  - ✅ 时间序列预测任务，更贴近实际应用
  - ✅ 论文刚发表，可直接对比
- **劣势**: 需要从HuggingFace下载，数据量大

### 3. FCMR (推荐⭐)
- **论文**: "FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning"
- **会议**: arXiv 2024 (最新版本: 2025-05-30)
- **数据集**: https://github.com/HYU-NLP/FCMR
- **特点**:
  - 金融跨模态多跳推理基准
  - 三个难度级别：Easy, Medium, Hard
  - Hard级别需要精确的三跳跨模态推理
  - 数据规模：757 Easy + 728 Medium + 714 Hard = 2,199样本
- **模态**: 
  1. 文本报告（Text）
  2. 财务表格（Table）
  3. 图表（Chart）
- **任务类型**: 多选分类（0-3个正确答案）
- **优势**: 
  - ✅ 专门设计用于评估跨模态多跳推理能力
  - ✅ 难度分级，可以评估模型在不同复杂度下的表现
  - ✅ 避免数据污染（使用真实金融数据，非Wikipedia）
  - ✅ 论文详细分析了模型失败原因，有深度分析
  - ✅ 最新论文，可直接对比
- **劣势**: 样本数量相对较少

## 🎯 推荐数据集选择

### 场景1: 想快速发表论文，需要直接可比较的baseline
**推荐: FinMultiTime**
- ✅ 论文刚发表（NeurIPS 2025），有现成的baseline结果
- ✅ 数据规模大，结果更有说服力
- ✅ 四模态设计，可以展示量子融合的优势
- ✅ 时间序列预测任务，应用价值高

### 场景2: 想展示跨模态推理能力
**推荐: FCMR**
- ✅ 专门设计用于多跳推理评估
- ✅ 有难度分级，可以展示模型在不同复杂度下的表现
- ✅ 论文有详细的分析，可以引用其分析方法

### 场景3: 想保持现有工作连续性
**推荐: 继续使用FinMME**
- ✅ 已经集成，可以直接运行
- ✅ 可以在此基础上添加新数据集进行对比

## 📝 论文引用格式

### FinMultiTime
```bibtex
@article{xu2025finmultitime,
  title={FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis},
  author={Xu, Wenyan and Xiang, Dawei and Liu, Yue and Wang, Xiyu and Ma, Yanxiang and Zhang, Liang and Xu, Chang and Zhang, Jiaheng},
  journal={arXiv preprint arXiv:2506.05019},
  year={2025}
}
```

### FCMR
```bibtex
@article{kim2024fcmr,
  title={FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning},
  author={Kim, Seunghee and Kim, Changhyeon and Kim, Taeuk},
  journal={arXiv preprint arXiv:2412.12567},
  year={2024}
}
```

### FinMME
```bibtex
@inproceedings{luo2025finmme,
  title={FinMME: Benchmark Dataset for Financial Multi-Modal Reasoning Evaluation},
  author={Luo, Junyu and others},
  booktitle={Proceedings of ACL 2025},
  year={2025}
}
```

## 🔄 数据集切换指南

### 切换到FinMultiTime

1. **下载数据集**:
```bash
# 方式1: 使用HuggingFace
from datasets import load_dataset
dataset = load_dataset("Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting")

# 方式2: 手动下载到 data/finmultitime/
```

2. **更新配置文件** (`configs/config_finmultitime.yaml`):
```yaml
data:
  source: finmultitime  # 新数据集
  data_dir: data/finmultitime
  market: SP500  # 或 HS300
  n_modalities: 4  # 四模态
  seq_lengths: [1, 1, 1, 1]
  feature_dims: [768, 768, 768, 768]
  output_dim: 1  # 回归任务
```

3. **运行训练**:
```bash
python train.py --config configs/config_finmultitime.yaml
```

### 切换到FCMR

1. **下载数据集**:
```bash
git clone https://github.com/HYU-NLP/FCMR.git
# 或手动下载到 data/fcmr/
```

2. **更新配置文件** (`configs/config_fcmr.yaml`):
```yaml
data:
  source: fcmr
  data_dir: data/fcmr
  difficulty: all  # easy, medium, hard, all
  n_modalities: 3
  seq_lengths: [1, 1, 1]
  feature_dims: [768, 768, 768]
  output_dim: 8  # 多标签分类（8个可能答案）
```

3. **运行训练**:
```bash
python train.py --config configs/config_fcmr.yaml
```

## 📊 与论文baseline对比

### FinMultiTime论文中的baseline结果

论文在HS300和S&P 500上测试了以下模型：
- RNN, LSTM, GRU, CNN
- TimesNet
- Transformer

**最佳结果** (35 stocks, 所有模态):
- Transformer: R² ≈ 0.97
- LSTM: R² ≈ 0.84
- GRU: R² ≈ 0.83

**你的量子模型目标**: 超越Transformer的0.97 R²，或展示在特定场景下的优势

### FCMR论文中的baseline结果

**Hard级别准确率**:
- Claude 3.5 Sonnet: 30.4%
- GPT-4o: 24.4%
- Gemini 1.5 Pro: 22.3%
- 随机选择: 12.3%

**你的量子模型目标**: 在Hard级别上超越30.4%，或展示在特定推理任务上的优势

## 🚀 快速开始

### 使用FinMultiTime
```bash
# 1. 安装依赖
pip install datasets transformers torch

# 2. 下载数据（自动）
python train.py --config configs/config_finmultitime.yaml

# 3. 训练所有模型
python run_all.py --config configs/config_finmultitime.yaml
```

### 使用FCMR
```bash
# 1. 下载数据
git clone https://github.com/HYU-NLP/FCMR.git data/fcmr

# 2. 训练
python train.py --config configs/config_fcmr.yaml
```

## 📈 论文写作建议

### 数据集部分
1. **介绍数据集选择理由**:
   - FinMultiTime: 规模最大、最新、四模态对齐
   - FCMR: 专门设计用于跨模态推理评估

2. **对比现有数据集**:
   - 与FinMME对比：规模、模态数量、任务类型
   - 与Time-MMD、CiK等对比：覆盖范围、数据质量

3. **数据预处理**:
   - 特征提取方法（BERT/ViT）
   - 数据对齐方式
   - 训练/验证/测试集划分

### 实验结果部分
1. **与论文baseline对比**:
   - 表格对比：你的模型 vs 论文中的baseline
   - 可视化：性能提升的图表

2. **消融实验**:
   - 不同模态组合的效果
   - 量子层的作用
   - 不同难度级别的表现（FCMR）

3. **分析讨论**:
   - 量子模型在哪些场景下表现更好
   - 计算复杂度分析
   - 实际应用价值

## 🔗 相关资源

- **FinMultiTime**: 
  - 论文: https://arxiv.org/html/2506.05019v1
  - 数据集: https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting
  
- **FCMR**: 
  - 论文: https://arxiv.org/pdf/2412.12567
  - 代码: https://github.com/HYU-NLP/FCMR

- **FinMME**: 
  - 数据集: https://huggingface.co/datasets/luojunyu/FinMME
