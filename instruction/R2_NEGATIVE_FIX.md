# R²为负数的原因和解决方案

## 📊 什么是R²？

**R²（决定系数 / Coefficient of Determination）** 衡量模型对数据的拟合程度。

### R²的计算公式

```python
R² = 1 - (SS_res / SS_tot)

其中：
- SS_res = Σ(y_true - y_pred)²  # 残差平方和
- SS_tot = Σ(y_true - y_mean)²  # 总平方和
- y_mean = mean(y_true)         # 真实值的均值
```

### R²的正常范围

- **R² = 1.0**：完美拟合，预测完全准确
- **R² = 0.5**：模型解释了50%的方差
- **R² = 0.0**：模型和预测均值一样差
- **R² < 0.0**：模型比预测均值还差（这是你的情况）

## ⚠️ R²为负数的原因

### 原因1：模型未训练或训练失败
**最常见的原因**：模型输出是随机的或未收敛

```python
# 症状：
# - 训练刚开始时R²为负
# - 模型参数未更新
# - 损失值非常大

# 解决方案：
# 1. 检查训练是否正常运行
# 2. 查看损失曲线是否下降
# 3. 等待训练完成
```

### 原因2：数据问题

**问题表现**：
- 标签数据全为0或常数
- 标签范围异常（如全为相同值）
- 数据未正确标准化

```python
# 检查数据：
import numpy as np

# 检查标签分布
print(f"Label mean: {np.mean(y_true)}")
print(f"Label std: {np.std(y_true)}")
print(f"Label min: {np.min(y_true)}, max: {np.max(y_true)}")
print(f"Unique values: {len(np.unique(y_true))}")

# 检查预测值
print(f"Pred mean: {np.mean(y_pred)}")
print(f"Pred std: {np.std(y_pred)}")
```

### 原因3：模型输出问题

**可能情况**：
- 模型输出维度不匹配
- 激活函数选择不当
- 输出层未正确初始化

```python
# 检查模型输出：
outputs = model(*modalities)
print(f"Output shape: {outputs.shape}")
print(f"Output range: {outputs.min().item()} - {outputs.max().item()}")
print(f"Expected label shape: {labels.shape}")
```

### 原因4：合成数据问题

在你的代码中，使用了随机生成的数据：

```python
# utils/data_loader.py 第82行
labels = np.random.randn(n_samples, output_dim)
```

**问题**：标签是完全随机的，与模态数据无关！

## 🔧 解决方案

### 方案1：改进合成数据生成（推荐）

让标签与模态数据相关：

```python
def generate_synthetic_data(n_samples=1000, n_modalities=3, seq_lengths=None, feature_dims=None, output_dim=1):
    """
    Generate synthetic multimodal data for testing
    """
    if seq_lengths is None:
        seq_lengths = [10] * n_modalities
    if feature_dims is None:
        feature_dims = [32] * n_modalities
    
    modalities = []
    for seq_len, feat_dim in zip(seq_lengths, feature_dims):
        mod = np.random.randn(n_samples, seq_len, feat_dim)
        modalities.append(mod)
    
    # ✅ 改进：让标签与模态数据相关
    # 方案A：简单线性组合
    labels = np.zeros((n_samples, output_dim))
    for mod in modalities:
        # 取每个模态的均值作为特征
        mod_feature = mod.mean(axis=(1, 2)).reshape(-1, 1)
        labels += mod_feature
    
    # 添加一些噪声
    labels += np.random.randn(n_samples, output_dim) * 0.1
    
    # 方案B：非线性组合
    # labels = np.sin(modalities[0].mean(axis=(1,2))) + np.cos(modalities[1].mean(axis=(1,2)))
    # labels = labels.reshape(-1, output_dim)
    
    return modalities, labels
```

### 方案2：使用真实数据集

使用FinMME数据集而不是合成数据：

```bash
# 下载FinMME数据集
python utils/download_finmme.py --output_dir data/finmme

# 使用FinMME配置训练
python train.py --config configs/config_finmme.yaml
```

### 方案3：调整训练参数

```yaml
# configs/config.yaml
training:
  epochs: 50  # 确保训练足够长时间
  learning_rate: 0.001  # 尝试调整学习率
  weight_decay: 0.0001
  early_stopping_patience: 10  # 增加耐心值
```

### 方案4：检查模型初始化

确保模型权重正确初始化：

```python
# 在模型初始化后检查
model = QuantumHybridModel(...)
print("Model initialized")

# 检查第一层权重
for name, param in model.named_parameters():
    print(f"{name}: mean={param.mean().item():.4f}, std={param.std().item():.4f}")
```

## 🧪 诊断步骤

### 步骤1：检查是否是训练初期

```python
# 训练初期R²为负是正常的
# 查看训练日志：
Epoch 1/50
  Train Loss: 2.5432, Val Loss: 2.6781
  Train R2: -0.5234, Val R2: -0.6123  # ⬅️ 初期可能为负

Epoch 5/50
  Train Loss: 0.8234, Val Loss: 0.9123
  Train R2: 0.2345, Val R2: 0.1892  # ⬅️ 应该逐渐变正

Epoch 20/50
  Train Loss: 0.2134, Val Loss: 0.3234
  Train R2: 0.7234, Val R2: 0.6543  # ⬅️ 最终应该较高
```

### 步骤2：检查数据

```python
# 添加到 train.py 的 main() 函数中
print("\n=== 数据检查 ===")
print(f"训练标签 - mean: {train_labels.mean():.4f}, std: {train_labels.std():.4f}")
print(f"验证标签 - mean: {val_labels.mean():.4f}, std: {val_labels.std():.4f}")
print(f"测试标签 - mean: {test_labels.mean():.4f}, std: {test_labels.std():.4f}")

# 检查模态数据
for i, mod in enumerate(train_mods):
    print(f"模态{i+1} - shape: {mod.shape}, mean: {mod.mean():.4f}, std: {mod.std():.4f}")
```

### 步骤3：监控训练过程

```python
# 在 train_epoch 中添加调试信息
def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch_idx, (modalities, labels) in enumerate(dataloader):
        modalities = [mod.to(device) for mod in modalities]
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(*modalities)
        
        # ✅ 添加调试信息
        if batch_idx == 0:  # 只打印第一个batch
            print(f"  Batch 0 - Output range: [{outputs.min():.4f}, {outputs.max():.4f}]")
            print(f"  Batch 0 - Label range: [{labels.min():.4f}, {labels.max():.4f}]")
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        all_preds.append(outputs.detach().cpu().numpy())
        all_labels.append(labels.detach().cpu().numpy())
    
    # ... rest of the function
```

## ✅ 推荐的修复方案

### 立即可用的修复

修改 `utils/data_loader.py`：

```python
def generate_synthetic_data(n_samples=1000, n_modalities=3, seq_lengths=None, feature_dims=None, output_dim=1):
    """
    Generate synthetic multimodal data for testing
    """
    if seq_lengths is None:
        seq_lengths = [10] * n_modalities
    if feature_dims is None:
        feature_dims = [32] * n_modalities
    
    modalities = []
    for seq_len, feat_dim in zip(seq_lengths, feature_dims):
        mod = np.random.randn(n_samples, seq_len, feat_dim)
        modalities.append(mod)
    
    # ✅ 改进的标签生成：让标签与模态数据相关
    labels = np.zeros((n_samples, output_dim))
    
    # 对每个模态提取简单特征并组合
    for i, mod in enumerate(modalities):
        # 取均值作为特征
        mod_mean = mod.mean(axis=(1, 2)).reshape(-1, 1)
        # 取标准差作为特征
        mod_std = mod.std(axis=(1, 2)).reshape(-1, 1)
        
        # 加权组合
        labels += 0.3 * mod_mean + 0.2 * mod_std
    
    # 添加小量噪声，模拟真实场景
    labels += np.random.randn(n_samples, output_dim) * 0.1
    
    # 标准化标签
    labels = (labels - labels.mean()) / (labels.std() + 1e-8)
    
    return modalities, labels
```

## 📝 验证修复

运行快速测试：

```bash
# 使用修复后的代码
python quick_test.py --config configs/config_quick.yaml
```

预期结果：
```
Epoch 1/3: Train Loss=0.8234, Val Loss=0.9123
  Epoch 1/3: Train Loss=0.8234, Val Loss=0.9123
Epoch 2/3: Train Loss=0.5123, Val Loss=0.6234
Epoch 3/3: Train Loss=0.3456, Val Loss=0.4567
✓ TFN passed! Time: 15.23s
```

R²应该在几个epoch后变为正数并逐渐提升。

## 🎯 总结

**R²为负的主要原因**：
1. ⚠️ **合成数据标签是随机的**，与模态数据无关（最可能）
2. 模型训练刚开始，还未收敛（正常情况）
3. 数据问题或模型问题（需要调试）
4. **模型初始化不当**，导致训练初期输出范围异常
5. **学习率过大**，导致训练不稳定
6. **Early stopping过早**，模型未充分训练

**解决方案**：
1. ✅ 修改 `generate_synthetic_data` 让标签与模态相关
2. ✅ 使用真实数据集（FinMME）
3. ✅ 增加训练轮数，观察R²是否提升
4. ✅ 添加调试信息监控训练过程
5. ✅ **改进模型初始化**（Xavier/Kaiming初始化）
6. ✅ **添加学习率调度器**（ReduceLROnPlateau）
7. ✅ **改进R²计算**，添加数据验证和调试信息
8. ✅ **添加数据标准化检查**

## 📚 参考资料

### 学术文献
1. **"When is R squared negative?"** - Stack Exchange Statistics
   - R²在测试集上可能为负，因为测试集均值与训练集不同
   - 负R²表示模型比简单均值预测还差

2. **"Explaining negative R-squared"** - Towards Data Science
   - 负R²发生在残差平方和超过总平方和时
   - 训练集上R²非负，但测试集上可能为负

3. **Neural Network Training Best Practices**
   - 学习率过大导致训练不稳定
   - 需要适当的权重初始化
   - 梯度裁剪和学习率调度很重要

### 技术文档
1. **scikit-learn r2_score文档**
   - R² = 1 - (SS_res / SS_tot)
   - 当SS_res > SS_tot时，R²为负
   - `force_finite=True`参数处理边界情况

2. **PyTorch训练最佳实践**
   - Xavier/Kaiming初始化
   - 学习率warmup
   - 梯度裁剪
   - 学习率调度器

### 常见问题
1. **Q: R²为负是否正常？**
   - A: 在测试集上是可能的，表示模型泛化能力差2. **Q: 如何修复负R²？**
   - A: 检查数据、改进模型、调整超参数、增加训练时间

3. **Q: 训练集R²为正，测试集R²为负？**
   - A: 典型的过拟合问题，需要正则化或更多数据