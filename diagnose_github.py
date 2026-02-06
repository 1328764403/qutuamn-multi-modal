"""
诊断 GitHub 连接问题并提供解决方案
"""

import subprocess
import sys
import os

def run_command(cmd, check=False):
    """运行 shell 命令"""
    try:
        result = subprocess.run(
            cmd,
            shell=True,
            check=check,
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        return result
    except Exception as e:
        return None

def main():
    print("=" * 60)
    print("GitHub 连接诊断工具")
    print("=" * 60)
    print()
    
    # 1. 检查当前 remote 配置
    print("1. 检查远程仓库配置...")
    result = run_command("git remote -v", check=False)
    if result and result.stdout:
        print(result.stdout)
        if "https://" in result.stdout:
            print("   ⚠️  当前使用 HTTPS，可能被网络阻止")
        elif "git@" in result.stdout:
            print("   ✓ 当前使用 SSH")
    print()
    
    # 2. 测试 HTTPS 连接
    print("2. 测试 HTTPS 连接...")
    result = run_command("curl -I https://github.com --max-time 5", check=False)
    if result and result.returncode == 0:
        print("   ✓ HTTPS 连接正常")
    else:
        print("   ✗ HTTPS 连接失败（可能被阻止）")
    print()
    
    # 3. 测试 SSH 连接（22端口）
    print("3. 测试 SSH 连接（22端口）...")
    result = run_command("ssh -T git@github.com -o ConnectTimeout=5", check=False)
    if result:
        if "successfully authenticated" in result.stdout.lower() or "Hi" in result.stdout:
            print("   ✓ SSH 22端口连接正常")
        elif "Connection timed out" in result.stderr or "Could not resolve" in result.stderr:
            print("   ✗ SSH 22端口连接失败")
        else:
            print(f"   ℹ SSH 响应: {result.stderr[:100]}")
    else:
        print("   ✗ SSH 22端口连接失败")
    print()
    
    # 4. 测试 SSH 连接（443端口）
    print("4. 测试 SSH 连接（443端口）...")
    result = run_command("ssh -T -p 443 git@ssh.github.com -o ConnectTimeout=5", check=False)
    if result:
        if "successfully authenticated" in result.stdout.lower() or "Hi" in result.stdout:
            print("   ✓ SSH 443端口连接正常")
        else:
            print(f"   ℹ SSH 443响应: {result.stderr[:100]}")
    else:
        print("   ✗ SSH 443端口连接失败")
    print()
    
    # 5. 检查 SSH 密钥
    print("5. 检查 SSH 密钥...")
    ssh_dir = os.path.expanduser("~/.ssh")
    if os.path.exists(ssh_dir):
        keys = [f for f in os.listdir(ssh_dir) if f.endswith(".pub")]
        if keys:
            print(f"   ✓ 找到 SSH 公钥: {', '.join(keys)}")
            print("   请确保这些密钥已添加到 GitHub:")
            print("   https://github.com/settings/keys")
        else:
            print("   ✗ 未找到 SSH 公钥")
    else:
        print("   ✗ .ssh 目录不存在")
    print()
    
    # 6. 提供解决方案
    print("=" * 60)
    print("解决方案")
    print("=" * 60)
    print()
    
    print("根据测试结果，推荐以下方案：")
    print()
    
    print("方案 1: 使用 SSH + 443端口（推荐，绕过防火墙）")
    print("-" * 60)
    print("1. 配置 SSH 使用 443 端口:")
    print()
    print("   创建/编辑文件: ~/.ssh/config")
    print("   添加以下内容:")
    print()
    print("   Host github.com")
    print("       Hostname ssh.github.com")
    print("       Port 443")
    print("       User git")
    print()
    print("2. 然后推送:")
    print("   git remote set-url origin git@github.com:1328764403/qutumn-train.git")
    print("   git push -u origin main")
    print()
    
    print("方案 2: 使用代理（如果有）")
    print("-" * 60)
    print("设置 Git 代理:")
    print("   git config --global http.proxy http://proxy.example.com:8080")
    print("   git config --global https.proxy http://proxy.example.com:8080")
    print()
    print("取消代理:")
    print("   git config --global --unset http.proxy")
    print("   git config --global --unset https.proxy")
    print()
    
    print("方案 3: 使用 GitHub CLI（gh）")
    print("-" * 60)
    print("安装: winget install GitHub.cli")
    print("登录: gh auth login")
    print("推送: gh repo sync")
    print()
    
    print("方案 4: 使用镜像站点（不推荐，但可用）")
    print("-" * 60)
    print("使用 Gitee 或其他 Git 托管平台作为中转")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
        sys.exit(1)
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
