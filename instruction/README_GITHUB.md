# 快速上传到 GitHub

## 🚀 一键上传（推荐）

### 第一次上传

```bash
# 1. 在 GitHub 上创建新仓库（不要初始化 README）

# 2. 在项目目录运行以下命令
cd quantum_multimodal_comparison

# 初始化 Git
git init
git add .
git commit -m "Initial commit"

# 连接到 GitHub（替换为你的仓库地址）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 推送
git branch -M main
git push -u origin main
```

### 后续更新

每次修改后，运行：

```bash
python utils/sync_to_github.py
```

或者手动：

```bash
git add .
git commit -m "描述你的修改"
git push
```

## 📖 详细指南

查看完整指南: [GITHUB_UPLOAD_GUIDE.md](GITHUB_UPLOAD_GUIDE.md)
