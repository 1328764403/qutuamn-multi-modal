# 解决 GitHub 网络连接问题

## 🔍 问题诊断

你遇到的错误：
```
Failed to connect to github.com port 443 after 21102 ms: Could not connect to server
```

这说明你的网络无法连接到 GitHub 的 443 端口（HTTPS）。

## 🚀 解决方案（按推荐顺序）

### 方案 1: 使用 SSH + 443 端口（最推荐）

这是最可靠的方案，可以绕过大多数防火墙限制。

#### 步骤 1: 运行自动配置脚本

```bash
python setup_ssh_443.py
```

脚本会自动：
- 检查/生成 SSH 密钥
- 配置 SSH 使用 443 端口

#### 步骤 2: 手动配置（如果脚本失败）

**创建/编辑文件**: `C:\Users\你的用户名\.ssh\config`

添加以下内容：

```
# GitHub over SSH using port 443 (bypasses firewall)
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/id_ed25519
```

#### 步骤 3: 确保 SSH 密钥已添加到 GitHub

1. **查看你的公钥**:
```bash
cat ~/.ssh/id_ed25519.pub
```

2. **添加到 GitHub**:
   - 访问: https://github.com/settings/keys
   - 点击 "New SSH key"
   - 粘贴公钥内容
   - 保存

#### 步骤 4: 测试连接

```bash
ssh -T git@github.com
```

如果看到 "Hi username! You've successfully authenticated..." 说明成功。

#### 步骤 5: 设置远程仓库并推送

```bash
git remote set-url origin git@github.com:1328764403/qutumn-train.git
git push -u origin main
```

### 方案 2: 使用代理（如果你有）

如果你有可用的代理服务器：

```bash
# 设置 HTTP 代理
git config --global http.proxy http://proxy.example.com:8080
git config --global https.proxy http://proxy.example.com:8080

# 推送
git push -u origin main

# 使用完后取消代理
git config --global --unset http.proxy
git config --global --unset https.proxy
```

### 方案 3: 使用 GitHub CLI

GitHub CLI 有时可以绕过网络限制：

```bash
# 安装 GitHub CLI
winget install GitHub.cli

# 登录（会打开浏览器）
gh auth login

# 推送（使用 gh 的认证）
git push -u origin main
```

### 方案 4: 使用 VPN 或科学上网工具

如果以上方法都不行，使用 VPN 或科学上网工具连接 GitHub。

## 🛠️ 诊断工具

运行诊断脚本查看具体问题：

```bash
python diagnose_github.py
```

## 📝 快速操作步骤（推荐）

```bash
# 1. 运行 SSH 配置脚本
python setup_ssh_443.py

# 2. 确保 SSH 密钥已添加到 GitHub
# （脚本会提示你）

# 3. 测试连接
ssh -T git@github.com

# 4. 设置远程仓库
git remote set-url origin git@github.com:1328764403/qutumn-train.git

# 5. 推送
git push -u origin main
```

## ❓ 常见问题

### Q: SSH 测试失败，提示 "Permission denied"

**A**: SSH 密钥未添加到 GitHub，或密钥不匹配。检查：
1. 公钥是否已添加到 GitHub
2. 使用的密钥文件是否正确

### Q: 443 端口也被阻止

**A**: 尝试：
1. 使用代理
2. 使用 VPN
3. 使用 GitHub CLI

### Q: 如何查看当前 remote 配置

**A**: 
```bash
git remote -v
```

### Q: 如何切换回 HTTPS

**A**:
```bash
git remote set-url origin https://github.com/1328764403/qutumn-train.git
```

## 🔐 SSH 密钥管理

### 生成新的 SSH 密钥

```bash
ssh-keygen -t ed25519 -C "your_email@example.com"
```

### 查看公钥

```bash
cat ~/.ssh/id_ed25519.pub
```

### 测试 SSH 连接

```bash
ssh -T git@github.com
```

## ✅ 验证配置

配置完成后，运行：

```bash
# 检查 remote
git remote -v

# 应该显示:
# origin  git@github.com:1328764403/qutumn-train.git (fetch)
# origin  git@github.com:1328764403/qutumn-train.git (push)

# 测试推送
git push -u origin main
```
