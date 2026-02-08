"""
FCMR 难度预测 — 小样本快速测试（一键运行）
使用约 80 条样本、小模型、6 个 epoch，几分钟内验证：数据加载 → 经典/量子模型训练 → 指标输出。
"""
import subprocess
import sys
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    config = root / "configs" / "config_fcmr_difficulty_quick.yaml"
    if not config.exists():
        print(f"Config not found: {config}")
        sys.exit(1)
    cmd = [sys.executable, "train.py", "--config", str(config)]
    print("Running:", " ".join(cmd))
    print("(FCMR difficulty prediction, max_samples=80, 3 models: TFN, LMF, QuantumHybrid)\n")
    subprocess.run(cmd, cwd=str(root))

if __name__ == "__main__":
    main()
