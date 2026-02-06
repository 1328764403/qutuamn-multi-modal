# 特征提取器模型本地下载指南

## 文件夹结构

模型需要下载到以下目录：

```
quantum_multimodal_comparison/
└── models/
    ├── bert-base-uncased/          # BERT 文本特征提取器
    │   ├── config.json
    │   ├── pytorch_model.bin (或 model.safetensors)
    │   ├── tokenizer_config.json
    │   ├── vocab.txt
    │   ├── tokenizer.json
    │   └── (其他文件)
    └── google-vit-base-patch16-224/  # ViT 图像特征提取器
        ├── config.json
        ├── pytorch_model.bin (或 model.safetensors)
        └── (其他文件)
```

## 下载方式

### 方式1: 使用 HuggingFace CLI（推荐）

```bash
# 安装 huggingface-cli（如果还没安装）
pip install huggingface-hub

# 下载 BERT 模型
huggingface-cli download bert-base-uncased --local-dir models/bert-base-uncased

# 下载 ViT 模型
huggingface-cli download google/vit-base-patch16-224 --local-dir models/google-vit-base-patch16-224
```

### 方式2: 使用 Python 脚本下载

```python
from huggingface_hub import snapshot_download

# 下载 BERT
snapshot_download(
    repo_id="bert-base-uncased",
    local_dir="models/bert-base-uncased"
)

# 下载 ViT
snapshot_download(
    repo_id="google/vit-base-patch16-224",
    local_dir="models/google-vit-base-patch16-224"
)
```

### 方式3: 使用镜像站点（中国用户推荐）

```bash
# 设置镜像
export HF_ENDPOINT=https://hf-mirror.com

# 然后使用方式1或方式2下载
```

**Windows PowerShell:**
```powershell
$env:HF_ENDPOINT='https://hf-mirror.com'
```

## 必需文件清单

### BERT 模型 (`models/bert-base-uncased/`)

**必需文件：**
- `config.json` - 模型配置文件
- `pytorch_model.bin` 或 `model.safetensors` - 模型权重文件
- `tokenizer_config.json` - Tokenizer 配置
- `vocab.txt` - 词汇表
- `tokenizer.json` - Tokenizer 文件

**可选文件：**
- `generation_config.json`
- `README.md`
- `tf_model.h5` (TensorFlow 格式，不需要)

### ViT 模型 (`models/google-vit-base-patch16-224/`)

**必需文件：**
- `config.json` - 模型配置文件
- `pytorch_model.bin` 或 `model.safetensors` - 模型权重文件

**可选文件：**
- `preprocessor_config.json`
- `README.md`
- `tf_model.h5` (TensorFlow 格式，不需要)

## 文件大小参考

- **BERT**: 约 440 MB
- **ViT**: 约 330 MB

## 验证下载

下载完成后，运行测试脚本验证：

```bash
python utils/test_feature_extractors.py --model-dir models
```

## 注意事项

1. **路径格式**: ViT 模型的路径中 `/` 需要替换为 `-`，即 `google/vit-base-patch16-224` → `google-vit-base-patch16-224`
2. **文件格式**: 优先使用 `model.safetensors`（更安全），如果没有则使用 `pytorch_model.bin`
3. **权限**: 确保有写入权限创建文件夹和文件
4. **磁盘空间**: 确保有至少 1GB 的可用空间
