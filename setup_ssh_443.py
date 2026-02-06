"""
自动配置 SSH 使用 443 端口连接 GitHub
解决网络连接问题
"""

import os
from pathlib import Path

def setup_ssh_config():
    """配置 SSH 使用 443 端口"""
    ssh_dir = Path.home() / ".ssh"
    config_file = ssh_dir / "config"
    
    # 创建 .ssh 目录（如果不存在）
    ssh_dir.mkdir(exist_ok=True)
    
    # 读取现有配置
    existing_config = ""
    if config_file.exists():
        existing_config = config_file.read_text(encoding='utf-8')
    
    # 检查是否已有 GitHub 配置
    if "Host github.com" in existing_config:
        print("⚠️  SSH config 中已存在 github.com 配置")
        response = input("是否覆盖? (y/n): ").strip().lower()
        if response != 'y':
            print("已取消")
            return False
        
        # 移除旧的 GitHub 配置
        lines = existing_config.split('\n')
        new_lines = []
        skip = False
        for line in lines:
            if line.strip().startswith("Host github.com"):
                skip = True
            elif skip and line.strip().startswith("Host "):
                skip = False
                new_lines.append(line)
            elif not skip:
                new_lines.append(line)
        
        existing_config = '\n'.join(new_lines)
    
    # 添加 GitHub 443 端口配置
    github_config = """
# GitHub over SSH using port 443 (bypasses firewall)
Host github.com
    Hostname ssh.github.com
    Port 443
    User git
    PreferredAuthentications publickey
    IdentityFile ~/.ssh/id_ed25519
    IdentityFile ~/.ssh/id_rsa
"""
    
    # 合并配置
    new_config = existing_config.strip() + github_config
    
    # 写入文件
    config_file.write_text(new_config, encoding='utf-8')
    
    print("✓ SSH 配置已更新")
    print(f"  配置文件: {config_file}")
    print()
    print("配置内容:")
    print(github_config)
    
    return True

def check_ssh_key():
    """检查 SSH 密钥"""
    ssh_dir = Path.home() / ".ssh"
    
    # 检查是否有公钥
    ed25519_pub = ssh_dir / "id_ed25519.pub"
    rsa_pub = ssh_dir / "id_rsa.pub"
    
    if ed25519_pub.exists():
        print(f"✓ 找到 SSH 密钥: {ed25519_pub.name}")
        print(f"  公钥路径: {ed25519_pub}")
        return True
    elif rsa_pub.exists():
        print(f"✓ 找到 SSH 密钥: {rsa_pub.name}")
        print(f"  公钥路径: {rsa_pub}")
        return True
    else:
        print("✗ 未找到 SSH 密钥")
        print()
        print("需要生成 SSH 密钥:")
        print("  ssh-keygen -t ed25519 -C \"your_email@example.com\"")
        return False

def main():
    print("=" * 60)
    print("配置 SSH 使用 443 端口连接 GitHub")
    print("=" * 60)
    print()
    
    # 检查 SSH 密钥
    if not check_ssh_key():
        print()
        response = input("是否现在生成 SSH 密钥? (y/n): ").strip().lower()
        if response == 'y':
            email = input("请输入你的邮箱: ").strip()
            if email:
                import subprocess
                subprocess.run(f'ssh-keygen -t ed25519 -C "{email}"', shell=True)
                print("\n✓ SSH 密钥已生成")
                print("\n请将公钥添加到 GitHub:")
                print("1. 访问: https://github.com/settings/keys")
                print("2. 点击 'New SSH key'")
                print("3. 复制以下公钥内容:")
                print()
                pub_key_path = Path.home() / ".ssh" / "id_ed25519.pub"
                if pub_key_path.exists():
                    print(pub_key_path.read_text(encoding='utf-8'))
            else:
                print("已取消")
                return
    
    print()
    
    # 配置 SSH
    if setup_ssh_config():
        print()
        print("=" * 60)
        print("配置完成！")
        print("=" * 60)
        print()
        print("下一步:")
        print("1. 确保 SSH 公钥已添加到 GitHub")
        print("2. 测试连接:")
        print("   ssh -T git@github.com")
        print("3. 设置远程仓库:")
        print("   git remote set-url origin git@github.com:1328764403/qutumn-train.git")
        print("4. 推送代码:")
        print("   git push -u origin main")
        print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
