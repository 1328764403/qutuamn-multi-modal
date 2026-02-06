# FinMultiTime本地数据设置指南

## 🔍 第一步：找到你的数据文件

运行检查脚本找到数据文件位置：

```bash
cd quantum_multimodal_comparison
python find_finmultitime.py
```

或者告诉我：
1. **数据文件在哪里？** (完整路径)
2. **文件格式是什么？** (.parquet, .csv, .json, .jsonl)
3. **文件名是什么？** (例如: `SP500_train.parquet` 或 `train.parquet`)

## 📁 支持的文件结构

### 方式1: 带市场前缀（推荐）
```
你的数据目录/
├── SP500_train.parquet
├── SP500_test.parquet
├── HS300_train.parquet
└── HS300_test.parquet
```

### 方式2: 不带市场前缀
```
你的数据目录/
├── train.parquet
└── test.parquet
```

### 方式3: 子目录结构
```
你的数据目录/
├── train/
│   └── data.parquet
└── test/
    └── data.parquet
```

## ⚙️ 第二步：配置数据路径

### 方法1: 修改配置文件

编辑 `configs/config_finmultitime.yaml`:

```yaml
data:
  source: finmultitime
  data_dir: "你的数据目录路径"  # 修改这里
  market: SP500  # 或 HS300，根据你的数据
  max_samples: 100  # 快速测试用100个样本
```

**示例路径**:
- Windows: `data_dir: "D:/datasets/finmultitime"`
- Windows: `data_dir: "C:/Users/你的用户名/data/finmultitime"`
- 相对路径: `data_dir: "data/finmultitime"`

### 方法2: 使用快速测试配置

编辑 `configs/config_quick_test_finmultitime.yaml`:

```yaml
data:
  source: finmultitime
  data_dir: "你的数据目录路径"  # 修改这里
  market: SP500
  max_samples: 100  # 限制100个样本用于快速测试
```

## ✅ 第三步：验证配置

运行快速测试：

```bash
python quick_test.py
```

选择选项2 (FinMultiTime)，如果数据加载成功会显示：
```
✓ 从本地加载了 X 条数据
```

## 🚨 常见问题

### Q: 找不到数据文件
A: 
1. 检查路径是否正确（注意Windows路径使用 `/` 或 `\\`）
2. 检查文件名是否匹配（train/test, SP500/HS300）
3. 运行 `python find_finmultitime.py` 查找数据

### Q: 数据格式不对
A: 
- 确保文件是 `.parquet`, `.csv`, `.json`, 或 `.jsonl` 格式
- 检查文件内容是否符合FinMultiTime数据格式要求

### Q: 想使用其他路径的数据
A: 
- 直接修改配置文件中的 `data_dir`
- 或创建新的配置文件，复制 `config_finmultitime.yaml` 并修改路径

## 📝 数据格式要求

FinMultiTime数据文件应包含以下列：

- `image_path` 或 `chart_path`: K线图路径
- `news_text` 或 `text`: 新闻文本
- `table_data` 或 `financial_table`: 财务表格（JSON格式）
- `time_series` 或 `price_series`: 时间序列（列表格式）
- `close_price` 或 `label` 或 `target`: 目标变量

## 🎯 快速开始

1. **找到数据**: `python find_finmultitime.py`
2. **配置路径**: 修改 `configs/config_quick_test_finmultitime.yaml` 中的 `data_dir`
3. **运行测试**: `python quick_test.py` 选择选项2
