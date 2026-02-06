"""
快速同步项目到 GitHub
自动执行 git add, commit, push
"""

import subprocess
import sys
import os
from pathlib import Path
from datetime import datetime

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
        print(f"错误: {e}")
        if e.stderr:
            print(f"错误信息: {e.stderr}")
        return None

def check_git_repo():
    """检查是否是 git 仓库"""
    result = run_command("git rev-parse --git-dir", check=False)
    return result and result.returncode == 0

def check_remote():
    """检查是否配置了远程仓库"""
    result = run_command("git remote -v", check=False)
    if result and result.returncode == 0 and result.stdout.strip():
        return True
    return False

def get_status():
    """获取 git 状态"""
    result = run_command("git status --short", check=False)
    if result and result.stdout.strip():
        return result.stdout
    return None

def main():
    """主函数"""
    print("=" * 60)
    print("GitHub 同步工具")
    print("=" * 60)
    print()
    
    # 检查是否是 git 仓库
    if not check_git_repo():
        print("❌ 当前目录不是 Git 仓库")
        print("\n请先初始化 Git 仓库:")
        print("  git init")
        print("  git remote add origin <你的GitHub仓库URL>")
        print("\n详细步骤请查看: GITHUB_UPLOAD_GUIDE.md")
        sys.exit(1)
    
    # 检查远程仓库
    if not check_remote():
        print("❌ 未配置远程仓库")
        print("\n请先添加远程仓库:")
        print("  git remote add origin <你的GitHub仓库URL>")
        print("\n例如:")
        print("  git remote add origin https://github.com/username/repo-name.git")
        sys.exit(1)
    
    # 显示当前状态
    print("📊 检查修改...")
    status = get_status()
    
    if not status:
        print("✓ 没有需要提交的修改")
        return
    
    print("\n修改的文件:")
    print(status)
    print()
    
    # 询问是否继续
    response = input("是否提交并推送到 GitHub? (y/n): ").strip().lower()
    if response != 'y':
        print("已取消")
        return
    
    # 获取提交信息
    print("\n请输入提交信息（描述本次修改）:")
    commit_message = input("> ").strip()
    
    if not commit_message:
        # 使用默认提交信息
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        commit_message = f"Update: {timestamp}"
        print(f"使用默认提交信息: {commit_message}")
    
    print("\n" + "=" * 60)
    print("开始同步...")
    print("=" * 60)
    
    # 1. 添加所有修改
    print("\n1. 添加文件...")
    result = run_command("git add .")
    if not result:
        print("❌ 添加文件失败")
        sys.exit(1)
    print("✓ 文件已添加")
    
    # 2. 提交
    print(f"\n2. 提交修改: {commit_message}")
    result = run_command(f'git commit -m "{commit_message}"')
    if not result:
        print("❌ 提交失败")
        sys.exit(1)
    print("✓ 已提交")
    
    # 3. 推送到 GitHub
    print("\n3. 推送到 GitHub...")
    result = run_command("git push")
    if not result:
        print("❌ 推送失败")
        print("\n可能的原因:")
        print("1. 网络连接问题")
        print("2. 认证失败（需要使用 Personal Access Token）")
        print("3. 远程仓库不存在或没有权限")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("✓ 同步完成！")
    print("=" * 60)
    print("\n你的代码已经成功推送到 GitHub")

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
