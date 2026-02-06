"""
修复 GitHub 远程仓库配置
"""

import subprocess
import sys

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
            print(result.stdout.strip())
        return result
    except subprocess.CalledProcessError as e:
        if e.stderr:
            print(f"错误: {e.stderr.strip()}")
        return None

def main():
    print("=" * 60)
    print("修复 GitHub 远程仓库配置")
    print("=" * 60)
    print()
    
    # 1. 查看当前 remote
    print("1. 查看当前远程仓库配置...")
    result = run_command("git remote -v", check=False)
    if result and result.stdout:
        print(result.stdout)
    print()
    
    # 2. 删除旧的 remote
    print("2. 删除旧的远程仓库配置...")
    result = run_command("git remote remove origin", check=False)
    if result and result.returncode == 0:
        print("✓ 已删除旧的远程仓库配置")
    else:
        print("ℹ 没有找到旧的远程仓库配置（可能已经删除）")
    print()
    
    # 3. 添加正确的 remote
    print("3. 添加正确的远程仓库...")
    github_url = input("请输入你的 GitHub 仓库 URL (例如: https://github.com/username/repo.git): ").strip()
    
    if not github_url:
        print("❌ 未输入 URL")
        return
    
    if not github_url.startswith("http"):
        print("❌ URL 格式不正确")
        return
    
    result = run_command(f'git remote add origin {github_url}')
    if result:
        print(f"✓ 已添加远程仓库: {github_url}")
    else:
        print("❌ 添加失败")
        return
    
    print()
    
    # 4. 验证配置
    print("4. 验证配置...")
    result = run_command("git remote -v", check=False)
    if result and result.stdout:
        print(result.stdout)
    
    print()
    print("=" * 60)
    print("配置完成！")
    print("=" * 60)
    print()
    print("现在可以推送代码了。")
    print()
    print("⚠️  重要：GitHub 需要使用 Personal Access Token 认证")
    print()
    print("如果推送时提示认证失败，请：")
    print("1. 访问: https://github.com/settings/tokens")
    print("2. 点击 'Generate new token (classic)'")
    print("3. 选择权限: 勾选 'repo'")
    print("4. 生成并复制 token")
    print("5. 推送时，用户名输入你的 GitHub 用户名")
    print("6. 密码输入刚才复制的 token（不是 GitHub 密码）")
    print()
    print("或者使用以下命令（替换 YOUR_TOKEN）:")
    print(f"  git push -u origin main")
    print()
    print("如果还是失败，可以尝试使用 SSH:")
    print("  git remote set-url origin git@github.com:username/repo.git")

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
