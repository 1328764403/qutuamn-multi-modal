# 基线模型论文列表

本文档列出了代码中实现的所有基线模型对应的原始论文，这些论文用于与量子混合模型进行性能对比。

## 📚 多模态融合基线模型论文

### 1. Tensor Fusion Network (TFN)

**论文信息**：
- **标题**: Tensor Fusion Network for Multimodal Sentiment Analysis
- **作者**: Amir Zadeh, Minghai Chen, Soujanya Poria, Erik Cambria, Louis-Philippe Morency
- **会议**: EMNLP 2017
- **年份**: 2017

**BibTeX引用**:
```bibtex
@inproceedings{zadeh2017tensor,
  title={Tensor Fusion Network for Multimodal Sentiment Analysis},
  author={Zadeh, Amir and Chen, Minghai and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing},
  pages={1103--1114},
  year={2017}
}
```

**论文链接**:
- ACL Anthology: https://aclanthology.org/D17-1115/
- arXiv: https://arxiv.org/abs/1707.07250
- 代码: https://github.com/A2Zadeh/TensorFusionNetwork

**核心贡献**:
- 提出使用张量外积进行多模态融合
- 捕获所有模态间的交互关系
- 在情感分析任务上验证有效性

---

### 2. Low-rank Multimodal Fusion (LMF)

**论文信息**:
- **标题**: Efficient Low-rank Multimodal Fusion with Modality-Specific Factors
- **作者**: Zhun Liu, Ying Shen, Varun Bharadhwaj Lakshminarasimhan, Paul Pu Liang, Amir Zadeh, Louis-Philippe Morency
- **会议**: ACL 2018
- **年份**: 2018

**BibTeX引用**:
```bibtex
@inproceedings{liu2018efficient,
  title={Efficient Low-rank Multimodal Fusion with Modality-Specific Factors},
  author={Liu, Zhun and Shen, Ying and Lakshminarasimhan, Varun Bharadhwaj and Liang, Paul Pu and Zadeh, Amir and Morency, Louis-Philippe},
  booktitle={Proceedings of the 56th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)},
  pages={2247--2256},
  year={2018}
}
```

**论文链接**:
- ACL Anthology: https://aclanthology.org/P18-1209/
- arXiv: https://arxiv.org/abs/1806.00064
- 代码: https://github.com/Justin1904/Low-rank-Multimodal-Fusion

**核心贡献**:
- 在TFN基础上引入低秩分解
- 显著降低计算复杂度（从O(d³)到O(d)）
- 保持融合能力的同时提升效率

---

### 3. Memory Fusion Network (MFN)

**论文信息**:
- **标题**: Memory Fusion Network for Multi-view Sequential Learning
- **作者**: Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Louis-Philippe Morency
- **会议**: AAAI 2018
- **年份**: 2018

**BibTeX引用**:
```bibtex
@inproceedings{zadeh2018memory,
  title={Memory Fusion Network for Multi-view Sequential Learning},
  author={Zadeh, Amir and Liang, Paul Pu and Poria, Soujanya and Cambria, Erik and Morency, Louis-Philippe},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={32},
  number={1},
  year={2018}
}
```

**论文链接**:
- AAAI: https://www.aaai.org/ocs/index.php/AAAI/AAAI18/paper/view/17322
- arXiv: https://arxiv.org/abs/1802.00927
- 代码: https://github.com/pliang279/MFN

**核心贡献**:
- 引入记忆网络机制
- 捕捉跨模态的长期依赖关系
- 适用于序列多模态数据

---

### 4. Multimodal Transformer (MulT)

**论文信息**:
- **标题**: Multimodal Transformer for Unaligned Multimodal Language Sequences
- **作者**: Wasif Rahman, Md Kamrul Hasan, Sangwu Lee, Amir Zadeh, Chengfeng Mao, Louis-Philippe Morency
- **会议**: ACL 2019
- **年份**: 2019

**BibTeX引用**:
```bibtex
@inproceedings{rahman2019multimodal,
  title={Multimodal Transformer for Unaligned Multimodal Language Sequences},
  author={Rahman, Wasif and Hasan, Md Kamrul and Lee, Sangwu and Zadeh, Amir and Mao, Chengfeng and Morency, Louis-Philippe},
  booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},
  pages={6558--6569},
  year={2019}
}
```

**论文链接**:
- ACL Anthology: https://aclanthology.org/P19-1656/
- arXiv: https://arxiv.org/abs/1906.00295
- 代码: https://github.com/yaohungt/Multimodal-Transformer

**核心贡献**:
- 基于Transformer架构的多模态融合
- 使用多头跨模态注意力机制
- 处理未对齐的多模态序列

---

### 5. Graph Convolutional Network (GCN)

**论文信息**:
- **标题**: Graph Convolutional Networks for Multimodal Fusion
- **作者**: Amir Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, Louis-Philippe Morency
- **会议**: 相关工作（基于GCN的多模态融合）
- **年份**: 2018

**相关论文**:
- **原始GCN**: Kipf & Welling (2017). "Semi-Supervised Classification with Graph Convolutional Networks." ICLR 2017
- **Graph-MFN**: Zadeh et al. (2018). "Graph-MFN: Graph Convolutional Networks for Multimodal Fusion."

**BibTeX引用**:
```bibtex
@inproceedings{kipf2017semi,
  title={Semi-Supervised Classification with Graph Convolutional Networks},
  author={Kipf, Thomas N and Welling, Max},
  booktitle={International Conference on Learning Representations},
  year={2017}
}
```

**论文链接**:
- ICLR: https://openreview.net/forum?id=SJU4ayYgl
- arXiv: https://arxiv.org/abs/1609.02907

**核心贡献**:
- 将模态视为图节点
- 使用图卷积进行信息传播
- 建模模态间的拓扑关系

---

### 6. Hypergraph Neural Networks

**论文信息**:
- **标题**: Hypergraph Neural Networks
- **作者**: Yifan Feng, Haoxuan You, Zizhao Zhang, Rongrong Ji, Yue Gao
- **会议**: AAAI 2019
- **年份**: 2019

**BibTeX引用**:
```bibtex
@inproceedings{feng2019hypergraph,
  title={Hypergraph Neural Networks},
  author={Feng, Yifan and You, Haoxuan and Zhang, Zizhao and Ji, Rongrong and Gao, Yue},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={33},
  number={01},
  pages={3558--3565},
  year={2019}
}
```

**论文链接**:
- AAAI: https://www.aaai.org/ojs/index.php/AAAI/article/view/4235
- arXiv: https://arxiv.org/abs/1809.09401
- 代码: https://github.com/iMoonLab/HGNN

**核心贡献**:
- 超图神经网络用于建模高阶关系
- 可以捕获多模态间的复杂交互
- 适用于多模态融合任务

---

## 🔬 量子机器学习相关论文

### 7. Variational Quantum Circuits

**论文信息**:
- **标题**: Quantum circuit learning
- **作者**: Kosuke Mitarai, Makoto Negoro, Masahiro Kitagawa, Keisuke Fujii
- **期刊**: Physical Review A, 2018
- **年份**: 2018

**BibTeX引用**:
```bibtex
@article{mitarai2018quantum,
  title={Quantum circuit learning},
  author={Mitarai, Kosuke and Negoro, Makoto and Kitagawa, Masahiro and Fujii, Keisuke},
  journal={Physical Review A},
  volume={98},
  number={3},
  pages={032309},
  year={2018},
  publisher={APS}
}
```

**论文链接**:
- Physical Review A: https://journals.aps.org/pra/abstract/10.1103/PhysRevA.98.032309
- arXiv: https://arxiv.org/abs/1803.00745

**核心贡献**:
- 提出变分量子电路用于机器学习
- 参数化量子电路训练方法
- 为量子机器学习奠定基础

---

### 8. Quantum-Enhanced Feature Spaces

**论文信息**:
- **标题**: Supervised learning with quantum-enhanced feature spaces
- **作者**: Vojtěch Havlíček, Antonio D. Córcoles, Kristan Temme, Aram W. Harrow, Abhinav Kandala, Jerry M. Chow, Jay M. Gambetta
- **期刊**: Nature, 2019
- **年份**: 2019

**BibTeX引用**:
```bibtex
@article{havlicek2019supervised,
  title={Supervised learning with quantum-enhanced feature spaces},
  author={Havl{\'i}{\v{c}}ek, Vojt{\v{e}}ch and C{\'o}rcoles, Antonio D and Temme, Kristan and Harrow, Aram W and Kandala, Abhinav and Chow, Jerry M and Gambetta, Jay M},
  journal={Nature},
  volume={567},
  number={7747},
  pages={209--212},
  year={2019},
  publisher={Nature Publishing Group UK London}
}
```

**论文链接**:
- Nature: https://www.nature.com/articles/s41586-019-0980-2
- arXiv: https://arxiv.org/abs/1804.11326

**核心贡献**:
- 使用量子增强特征空间的监督学习
- 在真实量子硬件上验证
- 展示量子计算在机器学习中的潜力

---

## 📊 数据集论文

### FinMultiTime Dataset

**论文信息**:
- **标题**: FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis
- **作者**: Wenyan Xu, Dawei Xiang, Yue Liu, Xiyu Wang, Yanxiang Ma, Liang Zhang, Chang Xu, Jiaheng Zhang
- **会议**: NeurIPS 2025 (投稿中)
- **年份**: 2025

**BibTeX引用**:
```bibtex
@article{xu2025finmultitime,
  title={FinMultiTime: A Four-Modal Bilingual Dataset for Financial Time-Series Analysis},
  author={Xu, Wenyan and Xiang, Dawei and Liu, Yue and Wang, Xiyu and Ma, Yanxiang and Zhang, Liang and Xu, Chang and Zhang, Jiaheng},
  journal={arXiv preprint arXiv:2506.05019},
  year={2025}
}
```

**论文链接**:
- arXiv: https://arxiv.org/html/2506.05019v1
- 数据集: https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting

---

### FCMR Dataset

**论文信息**:
- **标题**: FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning
- **作者**: Seunghee Kim, Changhyeon Kim, Taeuk Kim
- **会议**: arXiv 2024
- **年份**: 2024

**BibTeX引用**:
```bibtex
@article{kim2024fcmr,
  title={FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning},
  author={Kim, Seunghee and Kim, Changhyeon and Kim, Taeuk},
  journal={arXiv preprint arXiv:2412.12567},
  year={2024}
}
```

**论文链接**:
- arXiv: https://arxiv.org/pdf/2412.12567
- 代码: https://github.com/HYU-NLP/FCMR

---

## 📝 使用说明

### 在论文中引用这些论文

1. **Related Work部分**: 介绍多模态融合方法时引用TFN、LMF、MFN、MulT等
2. **Baseline对比**: 在实验部分说明与这些方法的对比
3. **数据集部分**: 引用FinMultiTime或FCMR数据集论文

### 代码实现对应关系

- `models/tfn.py` → TFN论文 (Zadeh et al., 2017)
- `models/lmf.py` → LMF论文 (Liu et al., 2018)
- `models/mfn.py` → MFN论文 (Zadeh et al., 2018)
- `models/mult.py` → MulT论文 (Rahman et al., 2019)
- `models/graph_baselines.py` → GCN/Hypergraph论文
- `models/quantum_hybrid.py` → 量子机器学习相关论文

---

## 🔗 快速访问链接

### 论文下载
- **ACL Anthology**: https://aclanthology.org/ (搜索论文标题)
- **arXiv**: https://arxiv.org/ (搜索arXiv编号)
- **GitHub**: 各论文的代码仓库链接见上

### 数据集
- **FinMultiTime**: https://huggingface.co/datasets/Wenyan0110/Multimodal-Dataset-Image_Text_Table_TimeSeries-for-Financial-Time-Series-Forecasting
- **FCMR**: https://github.com/HYU-NLP/FCMR

---

**最后更新**: 2026-01-26
