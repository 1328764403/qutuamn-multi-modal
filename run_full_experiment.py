"""
完整实验运行脚本
运行所有模型的完整训练，生成论文所需的所有结果和图表
"""

import os
import sys
import argparse
import subprocess
import time
from pathlib import Path


def run_command(cmd, description):
    """运行命令并显示进度"""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {cmd}")
    print()
    
    start_time = time.time()
    result = subprocess.run(cmd, shell=True, capture_output=False)
    elapsed_time = time.time() - start_time
    
    if result.returncode == 0:
        print(f"\n✓ {description} completed in {elapsed_time:.2f}s")
        return True
    else:
        print(f"\n✗ {description} failed with return code {result.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Run full experiment pipeline')
    parser.add_argument('--config', type=str, default='configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--device', type=str, default='cuda' if True else 'cpu',
                       help='Device to use')
    parser.add_argument('--skip_training', action='store_true',
                       help='Skip training, only generate tables')
    parser.add_argument('--skip_quick_test', action='store_true',
                       help='Skip quick test')
    args = parser.parse_args()
    
    print("="*60)
    print("FULL EXPERIMENT PIPELINE")
    print("="*60)
    print(f"Config: {args.config}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Step 1: Quick test (optional)
    if not args.skip_quick_test:
        print("\n[Step 1/4] Running quick test to verify all models...")
        quick_test_cmd = f"python quick_test.py --config {args.config} --device {args.device}"
        if not run_command(quick_test_cmd, "Quick Test"):
            print("Warning: Quick test failed, but continuing...")
            response = input("Continue with full training? (y/n): ")
            if response.lower() != 'y':
                print("Aborted.")
                return 1
    else:
        print("\n[Step 1/4] Skipping quick test (--skip_quick_test)")
    
    # Step 2: Full training
    if not args.skip_training:
        print("\n[Step 2/4] Running full training...")
        train_cmd = f"python train.py --config {args.config} --device {args.device}"
        if not run_command(train_cmd, "Full Training"):
            print("Error: Training failed!")
            return 1
    else:
        print("\n[Step 2/4] Skipping training (--skip_training)")
    
    # Step 3: Generate comparison plots
    print("\n[Step 3/4] Generating comparison plots...")
    compare_cmd = "python compare.py --results_dir results"
    if not run_command(compare_cmd, "Comparison Plots"):
        print("Warning: Comparison plot generation failed")
    
    # Step 4: Generate paper tables
    print("\n[Step 4/4] Generating paper tables...")
    table_cmd = "python generate_paper_tables.py --results_dir results --output_dir paper_tables"
    if not run_command(table_cmd, "Paper Tables"):
        print("Warning: Paper table generation failed")
    
    # Summary
    print("\n" + "="*60)
    print("EXPERIMENT PIPELINE COMPLETED")
    print("="*60)
    print("\nGenerated files:")
    print("  - results/all_results.json (all model results)")
    print("  - results/comparison_table.csv (comparison table)")
    print("  - results/comparison_bar.png (bar chart)")
    print("  - results/comparison_radar.png (radar chart)")
    print("  - paper_tables/comparison_table.tex (LaTeX table)")
    print("  - paper_tables/comparison_table.md (Markdown table)")
    print("  - paper_tables/comparison_summary.md (summary)")
    print("\nNext steps:")
    print("  1. Review results/all_results.json")
    print("  2. Check paper_tables/ for paper-ready tables")
    print("  3. Fill in EXPERIMENT_REPORT_TEMPLATE.md with your results")
    print("  4. Use REFERENCES.md for citations")
    print("="*60)
    
    return 0


if __name__ == '__main__':
    exit(main())

