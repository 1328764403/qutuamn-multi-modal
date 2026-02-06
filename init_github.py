"""
初始化 Git 仓库并连接到 GitHub
"""

import subprocess
import sys
import os
from pathlib import Path

def run_command(cmd, check=True):
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
        if result.stdout:
            print(result.stdout)
        return result
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(f"错误: {e.stderr}")
        return None

def check_git_installed():
    """检查 Git 是否安装"""
    result = run_command("git --version", check=False)
    return result and result.returncode == 0

def is_git_repo():
    """检查是否是 git 仓库"""
    result = run_command("git rev-parse --git-dir", check=False)
    return result and result.returncode == 0

def has_remote():
    """检查是否已有远程仓库"""
    result = run_command("git remote -v", check=False)
    if result and result.returncode == 0 and result.stdout.strip():
        return True
    return False

def main():
    """主函数"""
    print("=" * 60)
    print("GitHub 初始化工具")
    print("=" * 60)
    print()
    
    # 检查 Git 是否安装
    if not check_git_installed():
        print("❌ 未检测到 Git")
        print("\n请先安装 Git:")
        print("  Windows: https://git-scm.com/download/win")
        print("  或运行: winget install Git.Git")
        sys.exit(1)
    
    print("✓ Git 已安装")
    
    # 检查是否已经是 git 仓库
    if is_git_repo():
        print("✓ 已经是 Git 仓库")
        
        if has_remote():
            print("✓ 已配置远程仓库")
            print("\n如果要重新配置，请先运行:")
            print("  git remote remove origin")
            return
        
        # 已有仓库但没有远程
        print("\n检测到本地 Git 仓库，但未配置远程仓库")
        github_url = input("请输入你的 GitHub 仓库 URL: ").strip()
        
        if not github_url:
            print("❌ 未输入 URL")
            return
        
        if not github_url.startswith("http"):
            print("❌ URL 格式不正确，应该是: https://github.com/username/repo.git")
            return
        
        print(f"\n添加远程仓库: {github_url}")
        result = run_command(f'git remote add origin {github_url}')
        
        if result:
            print("✓ 远程仓库已添加")
            print("\n现在可以推送代码:")
            print("  git push -u origin main")
        else:
            print("❌ 添加远程仓库失败")
    else:
        # 初始化新仓库
        print("\n初始化 Git 仓库...")
        result = run_command("git init")
        
        if not result:
            print("❌ 初始化失败")
            sys.exit(1)
        
        print("✓ Git 仓库已初始化")
        
        # 添加文件
        print("\n添加文件...")
        result = run_command("git add .")
        if not result:
            print("❌ 添加文件失败")
            sys.exit(1)
        
        print("✓ 文件已添加")
        
        # 创建初始提交
        print("\n创建初始提交...")
        result = run_command('git commit -m "Initial commit: Quantum multimodal comparison project"')
        if not result:
            print("❌ 提交失败")
            print("提示: 如果是首次使用 Git，请先配置:")
            print("  git config --global user.name \"你的名字\"")
            print("  git config --global user.email \"你的邮箱\"")
            sys.exit(1)
        
        print("✓ 初始提交已创建")
        
        # 设置主分支
        print("\n设置主分支...")
        run_command("git branch -M main")
        print("✓ 主分支已设置")
        
        # 添加远程仓库
        print("\n" + "=" * 60)
        print("连接到 GitHub")
        print("=" * 60)
        print("\n请先在 GitHub 上创建新仓库:")
        print("1. 访问: https://github.com/new")
        print("2. 填写仓库名称")
        print("3. 不要勾选 'Initialize this repository with a README'")
        print("4. 点击 'Create repository'")
        print()
        
        github_url = input("请输入你的 GitHub 仓库 URL: ").strip()
        
        if not github_url:
            print("❌ 未输入 URL，稍后可以手动添加:")
            print("  git remote add origin <URL>")
            return
        
        if not github_url.startswith("http"):
            print("❌ URL 格式不正确")
            return
        
        print(f"\n添加远程仓库: {github_url}")
        result = run_command(f'git remote add origin {github_url}')
        
        if not result:
            print("❌ 添加远程仓库失败")
            return
        
        print("✓ 远程仓库已添加")
        
        # 推送到 GitHub
        print("\n" + "=" * 60)
        print("推送到 GitHub")
        print("=" * 60)
        print("\n准备推送到 GitHub...")
        response = input("是否现在推送? (y/n): ").strip().lower()
        
        if response == 'y':
            print("\n推送中...")
            result = run_command("git push -u origin main")
            
            if result:
                print("\n" + "=" * 60)
                print("✓ 成功！代码已上传到 GitHub")
                print("=" * 60)
            else:
                print("\n❌ 推送失败")
                print("\n可能的原因:")
                print("1. 需要认证（使用 Personal Access Token）")
                print("2. 网络连接问题")
                print("3. 仓库不存在或没有权限")
                print("\n可以稍后手动推送:")
                print("  git push -u origin main")
        else:
            print("\n可以稍后手动推送:")
            print("  git push -u origin main")
    
    print("\n" + "=" * 60)
    print("后续更新")
    print("=" * 60)
    print("\n每次修改后，运行以下命令同步:")
    print("  python utils/sync_to_github.py")
    print("\n或手动:")
    print("  git add .")
    print("  git commit -m \"描述修改\"")
    print("  git push")

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
