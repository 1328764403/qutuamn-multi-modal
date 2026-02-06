# 快速开始指南

## 🚀 三步快速开始

### 第一步：快速测试（5-10分钟）

验证所有模型是否能正常运行：

```bash
python quick_test.py --config configs/config_quick.yaml
```

**预期输出**:
- ✓ 所有模型通过测试
- 生成 `results/quick_test_results.json`

### 第二步：完整训练（根据硬件，几小时到一天）

运行完整实验：

```bash
python run_full_experiment.py --config configs/config.yaml
```

**或者分步运行**:

```bash
# 1. 训练所有模型
python train.py --config configs/config.yaml

# 2. 生成对比图
python compare.py --results_dir results

# 3. 生成论文表格
python generate_paper_tables.py --results_dir results
```

### 第三步：查看结果

所有结果在 `results/` 和 `paper_tables/` 目录：

- **对比表格**: `paper_tables/comparison_table.tex` (LaTeX格式，可直接插入论文)
- **对比图表**: `results/comparison_bar.png`, `results/comparison_radar.png`
- **详细结果**: `results/all_results.json`

## 📊 生成论文内容

### 1. 对比表格（LaTeX格式）

```bash
python generate_paper_tables.py --results_dir results
```

生成的文件：
- `paper_tables/comparison_table.tex` - 可直接复制到LaTeX论文
- `paper_tables/comparison_table.md` - Markdown格式
- `paper_tables/comparison_summary.md` - 实验总结

### 2. 填写实验报告

根据 `EXPERIMENT_REPORT_TEMPLATE.md` 填写你的实验结果。

### 3. 引用参考文献

参考 `REFERENCES.md` 中的论文列表和BibTeX格式引用。

## 📁 文件结构

```
quantum_multimodal_comparison/
├── quick_test.py              # 快速测试脚本
├── train.py                   # 完整训练脚本
├── run_full_experiment.py     # 完整实验流程
├── compare.py                 # 生成对比图
├── generate_paper_tables.py   # 生成论文表格
├── configs/
│   ├── config_quick.yaml      # 快速测试配置
│   └── config.yaml            # 完整实验配置
├── results/                   # 训练结果
│   ├── all_results.json       # 所有模型结果
│   ├── comparison_table.csv   # 对比表格
│   └── *.png                  # 各种图表
└── paper_tables/              # 论文表格
    ├── comparison_table.tex    # LaTeX表格
    ├── comparison_table.md    # Markdown表格
    └── comparison_summary.md  # 实验总结
```

## 🎯 使用场景

### 场景1: 快速验证代码
```bash
python quick_test.py
```

### 场景2: 完整实验（发论文用）
```bash
python run_full_experiment.py
```

### 场景3: 只训练特定模型
修改 `configs/config.yaml` 中的 `models_to_train`:
```yaml
models_to_train:
  - QuantumHybrid
  - TFN
```

### 场景4: 使用真实数据集
```bash
python train.py --config configs/config_finmme.yaml
```

## ⚙️ 配置说明

### 快速测试配置 (config_quick.yaml)
- 100个样本
- 3个epoch
- 小模型（hidden_dim=64）
- 2个量子比特

### 完整实验配置 (config.yaml)
- 1000个样本
- 50个epoch
- 标准模型（hidden_dim=128）
- 4个量子比特

## 📝 论文写作流程

1. **运行实验**
   ```bash
   python run_full_experiment.py
   ```

2. **生成表格**
   ```bash
   python generate_paper_tables.py
   ```

3. **复制LaTeX表格**
   - 打开 `paper_tables/comparison_table.tex`
   - 复制到你的LaTeX论文

4. **填写报告**
   - 打开 `EXPERIMENT_REPORT_TEMPLATE.md`
   - 根据 `results/all_results.json` 填写数据

5. **引用文献**
   - 参考 `REFERENCES.md`
   - 复制BibTeX格式引用

## 🔧 常见问题

**Q: 快速测试失败？**
- 检查依赖：`pip install -r requirements.txt`
- 检查配置文件路径

**Q: 训练太慢？**
- 使用 `config_quick.yaml` 进行快速测试
- 减少epochs数量
- 使用GPU：`--device cuda`

**Q: 如何只训练量子模型？**
- 修改配置文件中的 `models_to_train` 列表

## 📚 更多文档

- `README_EXPERIMENTS.md` - 详细实验指南
- `EXPERIMENT_REPORT_TEMPLATE.md` - 实验报告模板
- `REFERENCES.md` - 相关论文和引用
- `QUANTUM_MODEL_EXPLANATION.md` - 量子模型详解

## 🎉 完成！

运行完实验后，你将得到：
- ✅ 所有模型的训练结果
- ✅ 对比表格（LaTeX格式）
- ✅ 对比图表
- ✅ 实验报告模板

可以直接用于论文写作！

