# 数据集使用指南

## 当前数据集

项目目前使用**合成数据（Synthetic Data）**进行测试和演示。

### 合成数据特点

- **生成方式**: 使用 `np.random.randn()` 生成随机高斯分布数据
- **用途**: 快速测试模型框架，验证代码正确性
- **局限性**: 数据是随机的，没有真实的语义关系，不适合评估模型性能

### 合成数据配置

在 `configs/config.yaml` 中：

```yaml
data:
  n_samples: 1000          # 总样本数
  n_modalities: 3          # 模态数量
  seq_lengths: [10, 15, 12] # 每个模态的序列长度
  feature_dims: [32, 64, 48] # 每个模态的特征维度
  output_dim: 1           # 输出维度（回归任务）
```

## 使用真实数据集

### 支持的数据格式

1. **CSV文件**: 每个模态一个CSV文件
2. **NPZ文件**: NumPy压缩格式
3. **HDF5文件**: HDF5格式
4. **Pandas DataFrame**: 直接从DataFrame加载

### 数据格式要求

#### 多模态数据格式

每个模态应该是一个numpy数组：

```python
modalities = [
    np.array(...),  # 模态1: (n_samples, seq_len1, feature_dim1)
    np.array(...),  # 模态2: (n_samples, seq_len2, feature_dim2)
    np.array(...),  # 模态3: (n_samples, seq_len3, feature_dim3)
]
```

**注意**：
- 所有模态的 `n_samples` 必须相同
- `seq_len` 和 `feature_dim` 可以不同
- 如果没有时间维度，可以是 `(n_samples, feature_dim)`

#### 标签格式

```python
labels = np.array(...)  # (n_samples, output_dim)
```

- 回归任务: `output_dim = 1`
- 分类任务: `output_dim = num_classes`

### 示例：加载CMU-MOSI数据集

CMU-MOSI是一个经典的多模态情感分析数据集，包含文本、音频、视频三个模态。

```python
from utils.load_real_data import load_cmu_mosi_style

# 假设数据在 data/cmumosi/ 目录下
modalities, labels = load_cmu_mosi_style('data/cmumosi/')
```

数据目录结构：
```
data/cmumosi/
├── text_features.csv   # 文本特征
├── audio_features.csv  # 音频特征
├── video_features.csv  # 视频特征
└── labels.csv          # 标签（情感分数）
```

### 示例：加载自定义CSV数据

```python
from utils.load_real_data import load_from_csv

modalities, labels = load_from_csv(
    csv_paths=[
        'data/modality1.csv',
        'data/modality2.csv', 
        'data/modality3.csv'
    ],
    label_path='data/labels.csv'
)
```

### 示例：从NPZ文件加载

```python
import numpy as np
from utils.load_real_data import load_from_numpy

# 首先准备NPZ文件
modalities = [mod1, mod2, mod3]  # 你的模态数据
labels = your_labels

# 保存为NPZ
np.savez('data/multimodal_data.npz',
         modality_0=modalities[0],
         modality_1=modalities[1],
         modality_2=modalities[2],
         labels=labels)

# 加载
modalities, labels = load_from_numpy('data/multimodal_data.npz')
```

### 修改 train.py 使用真实数据

在 `train.py` 的 `main()` 函数中：

```python
def main():
    # ... 前面的代码 ...
    
    # 替换这部分：
    # modalities, labels = generate_synthetic_data(...)
    
    # 改为：
    from utils.load_real_data import load_from_csv
    modalities, labels = load_from_csv(
        csv_paths=['path/to/mod1.csv', 'path/to/mod2.csv', 'path/to/mod3.csv'],
        label_path='path/to/labels.csv'
    )
    
    # 确保数据形状正确
    print(f"Modality shapes: {[mod.shape for mod in modalities]}")
    print(f"Labels shape: {labels.shape}")
    
    # ... 后面的代码保持不变 ...
```

## 常见多模态数据集

### 1. CMU-MOSI / CMU-MOSEI
- **模态**: 文本、音频、视频
- **任务**: 情感分析（回归/分类）
- **下载**: http://multicomp.cs.cmu.edu/resources/cmu-mosi/

### 2. MIMIC-III
- **模态**: 文本（病历）、数值（生命体征）、图像（X光）
- **任务**: 医疗预测
- **下载**: https://mimic.mit.edu/

### 3. AV-Mnist
- **模态**: 图像、音频
- **任务**: 分类
- **下载**: https://github.com/ahmetgunduz/AV-MNIST

### 4. 自定义金融数据
- **模态**: 价格序列、宏观指标、文本新闻
- **任务**: 收益率预测

## 数据预处理建议

### 1. 标准化

```python
from sklearn.preprocessing import StandardScaler

scalers = []
for i, mod in enumerate(modalities):
    scaler = StandardScaler()
    # 如果是3D数据，需要reshape
    if len(mod.shape) == 3:
        mod_2d = mod.reshape(-1, mod.shape[-1])
        mod_scaled = scaler.fit_transform(mod_2d)
        modalities[i] = mod_scaled.reshape(mod.shape)
    else:
        modalities[i] = scaler.fit_transform(mod)
    scalers.append(scaler)
```

### 2. 缺失值处理

```python
import numpy as np

for i, mod in enumerate(modalities):
    # 用均值填充
    modalities[i] = np.nan_to_num(mod, nan=np.nanmean(mod))
```

### 3. 序列对齐

如果不同模态的序列长度不同，可以：

```python
# 方法1：截断到最短长度
min_len = min([mod.shape[1] for mod in modalities])
modalities = [mod[:, :min_len, :] for mod in modalities]

# 方法2：填充到最长长度
max_len = max([mod.shape[1] for mod in modalities])
# 使用零填充或插值
```

## 验证数据格式

运行以下代码验证你的数据格式是否正确：

```python
def validate_data(modalities, labels):
    """验证数据格式"""
    print("="*50)
    print("数据格式验证")
    print("="*50)
    
    # 检查模态数量
    print(f"模态数量: {len(modalities)}")
    
    # 检查每个模态
    n_samples = None
    for i, mod in enumerate(modalities):
        print(f"\n模态 {i+1}:")
        print(f"  形状: {mod.shape}")
        print(f"  数据类型: {mod.dtype}")
        print(f"  最小值: {mod.min():.4f}")
        print(f"  最大值: {mod.max():.4f}")
        print(f"  均值: {mod.mean():.4f}")
        print(f"  标准差: {mod.std():.4f}")
        
        if n_samples is None:
            n_samples = mod.shape[0]
        elif mod.shape[0] != n_samples:
            raise ValueError(f"模态 {i+1} 的样本数 ({mod.shape[0]}) 与其他模态不一致!")
    
    # 检查标签
    print(f"\n标签:")
    print(f"  形状: {labels.shape}")
    print(f"  数据类型: {labels.dtype}")
    print(f"  样本数: {labels.shape[0]}")
    
    if labels.shape[0] != n_samples:
        raise ValueError(f"标签样本数 ({labels.shape[0]}) 与模态样本数 ({n_samples}) 不一致!")
    
    print("\n✓ 数据格式验证通过!")
    return True

# 使用
modalities, labels = load_your_data(...)
validate_data(modalities, labels)
```

## 总结

- **当前**: 使用合成随机数据
- **生产环境**: 需要替换为真实数据集
- **数据格式**: 支持多种格式（CSV, NPZ, HDF5等）
- **关键要求**: 所有模态的样本数必须一致

如有问题，请参考 `utils/load_real_data.py` 中的示例代码。







