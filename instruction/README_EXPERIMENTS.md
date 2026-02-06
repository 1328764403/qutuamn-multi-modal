python# 实验运行指南

## 快速开始

### 1. 快速测试（验证代码能跑通）

使用小样本快速测试所有模型是否能正常运行：

```bash
python quick_test.py --config configs/config_quick.yaml
```

这个脚本会：
- 使用非常小的数据集（100个样本）
- 每个模型只训练3个epoch
- 验证所有模型是否能正常运行
- 生成测试报告：`results/quick_test_results.json`

**预期时间**: 5-10分钟

### 2. 完整实验（生成论文结果）

运行完整的实验，训练所有模型并生成对比结果：

```bash
# 方式1: 使用默认配置
python run_full_experiment.py

# 方式2: 指定配置和设备
python run_full_experiment.py --config configs/config.yaml --device cuda

# 方式3: 跳过快速测试，直接训练
python run_full_experiment.py --skip_quick_test
```

**预期时间**: 根据数据集大小和硬件，可能需要几小时到一天

### 3. 单独运行训练

如果只想训练模型，不生成对比表格：

```bash
python train.py --config configs/config.yaml --device cuda
```

### 4. 生成论文表格

如果已经训练完成，只需要生成对比表格：

```bash
# 生成所有格式的表格
python generate_paper_tables.py --results_dir results

# 只生成LaTeX表格
python generate_paper_tables.py --results_dir results --format latex

# 只生成Markdown表格
python generate_paper_tables.py --results_dir results --format markdown
```

## 配置文件说明

### config_quick.yaml
- **用途**: 快速测试
- **特点**: 
  - 小数据集（500样本）
  - 小模型（hidden_dim=64）
  - 少轮数（20 epochs）
  - 小量子比特数（2 qubits）

### config.yaml
- **用途**: 完整实验
- **特点**:
  - 标准数据集（1000样本）
  - 标准模型（hidden_dim=128）
  - 完整训练（50 epochs）
  - 标准量子比特数（4 qubits）

### config_finmme.yaml
- **用途**: 使用FinMME真实数据集
- **特点**: 使用真实金融多模态数据

## 输出文件说明

### 训练结果
- `results/all_results.json`: 所有模型的完整结果
- `results/[model]_best.pt`: 每个模型的最佳权重
- `results/[model]_losses.png`: 损失曲线图
- `results/[model]_r2.png`: R²分数曲线图

### 对比结果
- `results/comparison_table.csv`: CSV格式对比表
- `results/comparison_bar.png`: 柱状对比图
- `results/comparison_radar.png`: 雷达对比图

### 论文表格
- `paper_tables/comparison_table.tex`: LaTeX格式表格（可直接插入论文）
- `paper_tables/comparison_table.md`: Markdown格式表格
- `paper_tables/comparison_summary.md`: 实验总结
- `paper_tables/statistical_test.md`: 统计检验模板

## 实验流程

### 完整实验流程

```
1. 快速测试 (quick_test.py)
   ↓
2. 完整训练 (train.py)
   ↓
3. 生成对比图 (compare.py)
   ↓
4. 生成论文表格 (generate_paper_tables.py)
   ↓
5. 填写实验报告 (EXPERIMENT_REPORT_TEMPLATE.md)
```

### 论文写作流程

1. **运行实验**: 使用 `run_full_experiment.py`
2. **查看结果**: 检查 `results/all_results.json`
3. **生成表格**: 使用 `generate_paper_tables.py`
4. **填写报告**: 根据 `EXPERIMENT_REPORT_TEMPLATE.md` 填写
5. **引用文献**: 参考 `REFERENCES.md`

## 常见问题

### Q: 快速测试失败怎么办？
A: 检查：
1. 所有依赖是否安装（`pip install -r requirements.txt`）
2. 配置文件路径是否正确
3. 查看错误信息，可能是某个模型实现有问题

### Q: 训练时间太长怎么办？
A: 
1. 使用 `config_quick.yaml` 进行快速测试
2. 减少 `epochs` 数量
3. 减少模型大小（`hidden_dim`）
4. 使用GPU加速

### Q: 如何只训练特定模型？
A: 修改配置文件中的 `models_to_train` 列表：
```yaml
models_to_train:
  - QuantumHybrid
  - TFN
```

### Q: 如何添加新的评估指标？
A: 修改 `utils/metrics.py` 中的 `calculate_metrics` 函数

### Q: 如何保存更多训练信息？
A: 修改 `train.py` 中的 `train_model` 函数，添加更多日志

## 性能优化建议

1. **使用GPU**: 设置 `--device cuda`
2. **调整批次大小**: 根据GPU内存调整 `batch_size`
3. **减少量子比特数**: 如果量子模型太慢，减少 `n_qubits`
4. **使用混合精度**: 可以修改训练代码使用 `torch.cuda.amp`

## 结果解读

### 关键指标
- **R²**: 越高越好，表示模型解释的方差比例
- **RMSE**: 越低越好，均方根误差
- **MAE**: 越低越好，平均绝对误差
- **Best Epoch**: 模型达到最佳性能的轮数

### 模型对比
- 查看 `results/comparison_table.csv` 了解所有模型的性能
- 查看 `results/comparison_bar.png` 可视化对比
- 查看 `results/comparison_radar.png` 多维度对比

## 下一步

1. **分析结果**: 查看哪个模型表现最好
2. **生成论文表格**: 使用 `generate_paper_tables.py`
3. **填写实验报告**: 根据模板填写详细报告
4. **准备论文**: 使用生成的表格和报告撰写论文

## 联系和支持

如有问题，请检查：
1. 代码注释和文档
2. 错误日志
3. GitHub Issues（如果有）

