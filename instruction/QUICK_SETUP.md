# 快速设置指南 - 使用本地FinMultiTime数据

## ✅ 已配置好

配置文件已经设置为使用相对路径 `data/finmultitime`，数据加载器会自动查找数据文件。

## 📁 数据文件位置

将你的FinMultiTime数据文件放在以下位置之一：

### 选项1: 项目根目录下（推荐）
```
量经结合/
└── data/
    └── finmultitime/
        ├── SP500_train.parquet  (或 train.parquet)
        ├── SP500_test.parquet   (或 test.parquet)
        └── ...
```

### 选项2: quantum_multimodal_comparison目录下
```
quantum_multimodal_comparison/
└── data/
    └── finmultitime/
        ├── SP500_train.parquet
        └── ...
```

## 🚀 快速开始

### 1. 确认数据文件位置

确保你的数据文件在 `data/finmultitime/` 目录下，文件名格式为：
- `SP500_train.parquet` 或 `train.parquet`
- `SP500_test.parquet` 或 `test.parquet`

### 2. 运行快速测试（100个样本）

```bash
cd quantum_multimodal_comparison
python quick_test.py
```

选择选项2 (FinMultiTime)

### 3. 如果数据在其他位置

修改 `configs/config_quick_test_finmultitime.yaml`:

```yaml
data:
  data_dir: "你的数据目录路径"  # 例如: "D:/datasets/finmultitime"
```

## 📝 支持的文件格式

- `.parquet` (推荐，最快)
- `.csv`
- `.json`
- `.jsonl`

## 🔍 如果找不到数据

运行查找脚本：
```bash
python find_finmultitime.py
```

或者告诉我你的数据文件完整路径，我可以帮你配置。
