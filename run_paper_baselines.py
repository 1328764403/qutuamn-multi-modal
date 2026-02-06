"""
Paper-mode runner (no FinMultiTime):

- Default uses the original synthetic task (configs/config.yaml)
- Runs multiple seeds into separate subfolders
- Aggregates results (mean±std) and generates paper-ready tables
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baselines + quantum model for paper tables")
    p.add_argument("--config", type=str, default="configs/config.yaml", help="Config yaml path")
    p.add_argument("--device", type=str, default="cpu", help="cpu / cuda")
    p.add_argument("--seeds", type=str, default="42,43,44,45,46", help="Comma-separated seeds")
    p.add_argument("--out_root", type=str, default="paper_runs", help="Output root directory")
    p.add_argument("--skip_train", action="store_true", help="Skip training and only aggregate/tables")
    return p.parse_args()


def run(cmd: list[str]) -> int:
    print("Running:", " ".join(cmd))
    return subprocess.call(cmd)


def main() -> int:
    args = parse_args()
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    config_stem = Path(args.config).stem
    run_root = Path(args.out_root) / f"{config_stem}_{ts}"
    run_root.mkdir(parents=True, exist_ok=True)

    if not args.skip_train:
        for seed in seeds:
            seed_dir = run_root / f"seed_{seed}"
            seed_dir.mkdir(parents=True, exist_ok=True)
            code = run(
                [
                    sys.executable,
                    "train.py",
                    "--config",
                    args.config,
                    "--device",
                    args.device,
                    "--seed",
                    str(seed),
                    "--save_dir",
                    str(seed_dir),
                ]
            )
            if code != 0:
                print(f"[warn] seed={seed} failed (exit={code}). Continuing...")

    # Aggregate + tables (generate_paper_tables.py supports nested runs)
    tables_dir = run_root / "paper_tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    code = run(
        [
            sys.executable,
            "generate_paper_tables.py",
            "--results_dir",
            str(run_root),
            "--output_dir",
            str(tables_dir),
            "--format",
            "both",
        ]
    )
    if code != 0:
        return code

    print("\nDone.")
    print("Run root:", run_root)
    print("Tables:", tables_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

