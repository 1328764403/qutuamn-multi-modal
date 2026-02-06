# GitHub 上传指南

## 📋 准备工作

### 1. 安装 Git（如果还没安装）

**Windows:**
- 下载: https://git-scm.com/download/win
- 或使用: `winget install Git.Git`

**验证安装:**
```bash
git --version
```

### 2. 配置 Git（首次使用）

```bash
git config --global user.name "你的名字"
git config --global user.email "你的邮箱"
```

## 🚀 上传到 GitHub 的步骤

### 步骤 1: 在 GitHub 上创建新仓库

1. 登录 GitHub: https://github.com
2. 点击右上角 `+` → `New repository`
3. 填写仓库信息：
   - Repository name: `quantum-multimodal-comparison` (或你喜欢的名字)
   - Description: 量子多模态融合模型对比研究
   - 选择 Public 或 Private
   - **不要**勾选 "Initialize this repository with a README"
4. 点击 `Create repository`

### 步骤 2: 初始化本地 Git 仓库

在项目根目录（`quantum_multimodal_comparison`）打开终端：

```bash
# 进入项目目录
cd quantum_multimodal_comparison

# 初始化 Git 仓库
git init

# 添加所有文件（.gitignore 会自动排除不需要的文件）
git add .

# 创建初始提交
git commit -m "Initial commit: Quantum multimodal comparison project"
```

### 步骤 3: 连接到 GitHub 仓库

```bash
# 添加远程仓库（替换 YOUR_USERNAME 和 YOUR_REPO_NAME）
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git

# 例如：
# git remote add origin https://github.com/yourusername/quantum-multimodal-comparison.git
```

### 步骤 4: 推送到 GitHub

```bash
# 推送到 GitHub（首次推送）
git branch -M main
git push -u origin main
```

如果提示需要认证，GitHub 现在使用 Personal Access Token：
1. 访问: https://github.com/settings/tokens
2. 生成新 token (classic)
3. 选择权限: `repo`
4. 复制 token，在密码提示时使用

## 🔄 同步更新到 GitHub

### 方法 1: 使用提供的脚本（推荐）

运行同步脚本：

```bash
python utils/sync_to_github.py
```

### 方法 2: 手动同步

每次修改后，执行以下命令：

```bash
# 1. 查看修改的文件
git status

# 2. 添加修改的文件
git add .

# 3. 提交修改（写清楚修改内容）
git commit -m "描述你的修改内容"

# 4. 推送到 GitHub
git push
```

### 方法 3: 使用 Git GUI 工具

- **GitHub Desktop**: https://desktop.github.com
- **SourceTree**: https://www.sourcetreeapp.com
- **VS Code**: 内置 Git 支持

## 📝 提交信息规范

好的提交信息示例：

```bash
git commit -m "添加特征提取器下载脚本"
git commit -m "修复 BERT 模型加载问题"
git commit -m "更新 README 文档"
git commit -m "添加测试脚本"
```

## ⚠️ 注意事项

### 不会上传的文件（已在 .gitignore 中配置）

- 模型文件（.bin, .safetensors）- 太大
- 数据文件（data/）
- 缓存文件（__pycache__/）
- 结果文件（results/, *.pt, *.pth）
- 压缩文件（*.rar, *.zip）

### 如果需要上传模型文件

模型文件太大，GitHub 有 100MB 文件大小限制。建议：

1. **使用 Git LFS** (Large File Storage):
```bash
# 安装 Git LFS
git lfs install

# 跟踪大文件
git lfs track "*.bin"
git lfs track "*.safetensors"

# 然后正常提交
git add .gitattributes
git commit -m "Add Git LFS tracking"
```

2. **或使用外部存储**:
   - Google Drive
   - OneDrive
   - 在 README 中提供下载链接

## 🔧 常见问题

### 问题 1: 推送被拒绝

```bash
# 如果远程仓库有 README 等文件，先拉取
git pull origin main --allow-unrelated-histories

# 解决冲突后再次推送
git push -u origin main
```

### 问题 2: 认证失败

使用 Personal Access Token 而不是密码：
1. GitHub Settings → Developer settings → Personal access tokens
2. 生成新 token
3. 使用 token 作为密码

### 问题 3: 想撤销最后一次提交

```bash
# 撤销提交但保留修改
git reset --soft HEAD~1

# 完全撤销
git reset --hard HEAD~1
```

## 📚 更多资源

- Git 官方文档: https://git-scm.com/doc
- GitHub 指南: https://docs.github.com
- Git 教程: https://www.atlassian.com/git/tutorials
