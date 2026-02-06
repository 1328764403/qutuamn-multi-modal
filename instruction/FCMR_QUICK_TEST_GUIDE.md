# FCMR 数据集快速测试指南

本指南介绍如何使用真实小样本数据快速测试 FCMR 数据集加载和模型功能。

## 测试步骤

### 步骤 1: 数据加载测试（推荐先运行）

首先测试数据是否能正确加载：

```bash
cd quantum_multimodal_comparison
python test_fcmr_data_loading.py
```

这个脚本会：
- 测试每个难度级别（easy, medium, hard）的数据加载
- 测试加载所有难度级别的数据
- 检查数据质量和标签分布
- 不训练模型，只验证数据加载功能

**预期输出：**
- ✓ 每个难度级别的数据加载成功
- ✓ 显示数据形状和样本数
- ✓ 检查数据质量（NaN 值等）
- ✓ 显示标签分布

### 步骤 2: 完整快速测试

如果数据加载测试通过，运行完整测试（包括模型测试）：

```bash
python test_fcmr_quick.py
```

这个脚本会：
- 加载数据（使用 `configs/config_fcmr_quick_test.yaml` 配置）
- 创建数据加载器
- 测试模型前向传播（TFN, LMF, QuantumHybrid）
- 验证模型是否能正常工作

**预期输出：**
- ✓ 数据加载成功
- ✓ 数据加载器创建成功
- ✓ 每个模型的前向传播测试通过

### 步骤 3: 小样本训练测试（可选）

如果想测试完整的训练流程，可以运行：

```bash
python train.py --config configs/config_fcmr_quick_test.yaml
```

这个配置会：
- 使用 50 个样本（easy 难度级别）
- 训练 3 个模型（TFN, LMF, QuantumHybrid）
- 只训练 5 个 epoch
- 使用较小的模型规模

## 配置文件说明

### `configs/config_fcmr_quick_test.yaml`

快速测试配置文件，包含以下设置：

- **数据设置**:
  - `difficulty: easy` - 使用 easy 难度级别
  - `max_samples: 50` - 限制为 50 个样本
  - `batch_size: 8` - 小批次大小

- **模型设置**:
  - `hidden_dim: 128` - 较小的隐藏维度
  - 只训练 3 个模型（TFN, LMF, QuantumHybrid）

- **训练设置**:
  - `epochs: 5` - 只训练 5 个 epoch
  - `early_stopping_patience: 3` - 较小的耐心值

## 自定义测试

### 修改样本数量

编辑 `configs/config_fcmr_quick_test.yaml`:

```yaml
data:
  max_samples: 100  # 改为你想要的样本数
```

### 修改难度级别

```yaml
data:
  difficulty: medium  # 或 "hard", "all"
```

### 修改要测试的模型

```yaml
models_to_train:
  - TFN
  - LMF
  - QuantumHybrid
  # 可以添加更多模型
```

## 常见问题

### 1. 数据目录不存在

**错误信息：**
```
错误: 数据目录不存在: data/fcmr
```

**解决方法：**
- 确保 FCMR 数据集已下载
- 检查数据目录结构是否正确：
  ```
  data/fcmr/
  ├── dataset/
  │   ├── easy/
  │   │   ├── easy_data.csv
  │   │   ├── easy_test_table_modality/
  │   │   └── easy_test_text_modality_chunk/
  │   ├── medium/
  │   └── hard/
  ```

### 2. 文件未找到

**错误信息：**
```
✗ 文件未找到: ...
```

**解决方法：**
- 检查 CSV 文件是否存在：`dataset/{difficulty}/{difficulty}_data.csv`
- 检查表格和文本文件目录是否存在
- 确保文件命名正确

### 3. 内存不足

**解决方法：**
- 减少 `max_samples` 数量
- 设置 `use_pretrained_features: false`（不使用预训练模型）
- 减小 `batch_size`

### 4. 模型前向传播失败

**可能原因：**
- 数据形状不匹配
- 模型配置错误
- 设备问题（CUDA/CPU）

**解决方法：**
- 检查数据加载是否成功
- 检查模型配置是否正确
- 尝试使用 CPU：`python test_fcmr_quick.py --device cpu`

## 下一步

如果所有测试通过，可以：

1. **运行完整训练**:
   ```bash
   python train.py --config configs/config_fcmr.yaml
   ```

2. **运行所有模型比较**:
   ```bash
   python compare.py --config configs/config_fcmr.yaml
   ```

3. **生成实验结果**:
   ```bash
   python generate_paper_tables.py
   ```

## 测试时间估算

- **数据加载测试**: 1-2 分钟（取决于数据量和是否使用预训练模型）
- **完整快速测试**: 2-5 分钟（取决于模型数量和设备）
- **小样本训练测试**: 5-15 分钟（取决于模型数量和设备）

## 注意事项

1. **首次运行**：如果使用预训练模型（`use_pretrained_features: true`），首次运行需要下载模型，可能需要一些时间。

2. **数据量**：FCMR 数据集较大，建议先用小样本测试，确认一切正常后再运行完整训练。

3. **设备**：如果有 GPU，建议使用 GPU 加速。如果没有 GPU，可以设置 `use_pretrained_features: false` 来加快速度。

4. **内存**：如果内存不足，可以减少 `max_samples` 或 `batch_size`。
