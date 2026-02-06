# FinMME 数据集手动下载指南

如果自动下载失败（网络问题、代理问题等），可以使用以下方法手动下载。

## 方法1: 使用镜像站点（推荐）

### Windows PowerShell

```powershell
# 设置镜像环境变量
$env:HF_ENDPOINT='https://hf-mirror.com'

# 运行下载脚本
python utils/download_finmme.py --output_dir data/finmme --splits train test
```

### Windows CMD

```cmd
set HF_ENDPOINT=https://hf-mirror.com
python utils/download_finmme.py --output_dir data/finmme --splits train test
```

### Linux/Mac

```bash
export HF_ENDPOINT=https://hf-mirror.com
python utils/download_finmme.py --output_dir data/finmme --splits train test
```

## 方法2: 使用 datasets 库手动下载

```python
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'  # 使用镜像

from datasets import load_dataset

# 下载数据集
dataset = load_dataset("luojunyu/FinMME")

# 保存为本地文件
dataset['train'].to_parquet('data/finmme/train.parquet')
dataset['test'].to_parquet('data/finmme/test.parquet')
```

然后运行处理脚本：

```bash
python utils/download_finmme_manual.py data/finmme/train.parquet --output_dir data/finmme --split train
python utils/download_finmme_manual.py data/finmme/test.parquet --output_dir data/finmme --split test
```

## 方法3: 从网页手动下载

1. 访问镜像站点：
   - https://hf-mirror.com/datasets/luojunyu/FinMME

2. 下载以下文件：
   - `data/train-00000-of-00001.parquet`
   - `data/test-00000-of-00001.parquet`

3. 保存到本地目录，例如：
   ```
   downloads/
     train-00000-of-00001.parquet
     test-00000-of-00001.parquet
   ```

4. 使用处理脚本：

```bash
python utils/download_finmme_manual.py downloads/train-00000-of-00001.parquet --output_dir data/finmme --split train
python utils/download_finmme_manual.py downloads/test-00000-of-00001.parquet --output_dir data/finmme --split test
```

## 方法4: 使用 Git LFS

```bash
# 安装 Git LFS（如果还没安装）
git lfs install

# 克隆数据集仓库
git clone https://hf-mirror.com/datasets/luojunyu/FinMME.git data/finmme_raw

# 处理下载的文件
python utils/download_finmme_manual.py data/finmme_raw/data/train-00000-of-00001.parquet --output_dir data/finmme --split train
python utils/download_finmme_manual.py data/finmme_raw/data/test-00000-of-00001.parquet --output_dir data/finmme --split test
```

## 方法5: 使用代理

如果你有代理服务器：

### Windows PowerShell

```powershell
# 设置代理
$env:HTTP_PROXY='http://your-proxy:port'
$env:HTTPS_PROXY='http://your-proxy:port'

# 运行下载
python utils/download_finmme.py --output_dir data/finmme
```

### Linux/Mac

```bash
export HTTP_PROXY=http://your-proxy:port
export HTTPS_PROXY=http://your-proxy:port
python utils/download_finmme.py --output_dir data/finmme
```

## 验证下载

下载完成后，验证数据：

```bash
python utils/download_finmme.py --output_dir data/finmme --verify
```

或手动检查：

```python
import pandas as pd
from pathlib import Path

data_dir = Path('data/finmme')

# 检查文件
for split in ['train', 'test']:
    csv_path = data_dir / f'{split}.csv'
    parquet_path = data_dir / f'{split}.parquet'
    
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        print(f"{split}: {len(df)} 条数据")
        print(f"  列: {list(df.columns)}")
    else:
        print(f"{split}: 文件不存在")
```

## 常见问题

### Q: 下载很慢怎么办？

A: 
1. 使用镜像站点（hf-mirror.com）
2. 使用代理
3. 手动下载后使用处理脚本

### Q: 下载中断了怎么办？

A: 可以重新运行下载脚本，已下载的文件会被跳过。

### Q: 图像文件很大怎么办？

A: 图像文件会保存在 `data/finmme/train/images/` 和 `data/finmme/test/images/` 目录下。如果磁盘空间不足，可以考虑：
- 只下载训练集
- 使用较小的图像尺寸
- 不保存图像，直接使用特征

### Q: 如何只下载训练集？

A: 

```bash
python utils/download_finmme.py --output_dir data/finmme --splits train
```

## 下一步

下载完成后，可以开始训练：

```bash
python train.py --config configs/config_finmme.yaml
```







