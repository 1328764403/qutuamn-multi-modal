"""
生成论文用的对比表格
支持LaTeX格式和Markdown格式
"""

import json
import os
import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def _read_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_results(results_dir: str):
    """
    加载实验结果。

    - 若 results_dir 直接包含 all_results.json：按单次运行读取
    - 否则：递归搜索子目录下的 all_results.json（用于多 seed 多次运行），返回聚合结果
    """
    results_path = os.path.join(results_dir, "all_results.json")
    if os.path.exists(results_path):
        return _read_json(results_path)

    # Multi-run mode: find all all_results.json under results_dir
    root = Path(results_dir)
    json_files = sorted([str(p) for p in root.rglob("all_results.json")])
    if not json_files:
        raise FileNotFoundError(
            f"Results file not found: {results_path} (also no nested all_results.json found)"
        )

    runs = [_read_json(p) for p in json_files]
    return aggregate_runs(runs)


def aggregate_runs(runs: list[dict]) -> dict:
    """把多次运行（多 seed）的 all_results.json 聚合成 mean/std 结构。"""
    # collect: model -> metric_name -> list[float]
    agg: dict = {}
    for run in runs:
        for model_name, payload in run.items():
            agg.setdefault(model_name, {"test_metrics": {}, "val_metrics": {}, "n_runs": 0})
            test_metrics = payload.get("test_metrics", {})
            val_metrics = payload.get("val_metrics", {})

            for k, v in test_metrics.items():
                agg[model_name]["test_metrics"].setdefault(k, []).append(float(v))
            for k, v in val_metrics.items():
                agg[model_name]["val_metrics"].setdefault(k, []).append(float(v))

            agg[model_name]["n_runs"] += 1

    # finalize mean/std
    out: dict = {}
    for model_name, d in agg.items():
        out[model_name] = {
            "n_runs": int(d["n_runs"]),
            "test_metrics_mean": {},
            "test_metrics_std": {},
            "val_metrics_mean": {},
            "val_metrics_std": {},
        }
        for k, values in d["test_metrics"].items():
            arr = np.asarray(values, dtype=np.float64)
            out[model_name]["test_metrics_mean"][k] = float(arr.mean())
            out[model_name]["test_metrics_std"][k] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        for k, values in d["val_metrics"].items():
            arr = np.asarray(values, dtype=np.float64)
            out[model_name]["val_metrics_mean"][k] = float(arr.mean())
            out[model_name]["val_metrics_std"][k] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0

    return out


def generate_latex_table(results, output_path=None):
    """生成LaTeX格式的对比表格"""
    
    # 准备数据
    data = []
    for model_name, metrics in results.items():
        # Support both single-run and aggregated results
        if "test_metrics" in metrics:
            test_metrics = metrics["test_metrics"]
            val_metrics = metrics.get("val_metrics", {})
            r2 = test_metrics["R2"]
            rmse = test_metrics["RMSE"]
            mae = test_metrics["MAE"]
            mse = test_metrics["MSE"]
            val_r2 = val_metrics.get("R2", 0)
            best_epoch = metrics.get("best_epoch", "-")
        else:
            test_mean = metrics["test_metrics_mean"]
            test_std = metrics["test_metrics_std"]
            val_mean = metrics.get("val_metrics_mean", {})
            val_std = metrics.get("val_metrics_std", {})
            r2 = test_mean.get("R2", 0.0)
            rmse = test_mean.get("RMSE", 0.0)
            mae = test_mean.get("MAE", 0.0)
            mse = test_mean.get("MSE", 0.0)
            val_r2 = val_mean.get("R2", 0.0)
            best_epoch = f"{metrics.get('n_runs', 0)} runs"
        
        data.append({
            'Model': model_name,
            'R²': f"{r2:.4f}",
            'RMSE': f"{rmse:.4f}",
            'MAE': f"{mae:.4f}",
            'MSE': f"{mse:.4f}",
            'Val R²': f"{val_r2:.4f}",
            'Best Epoch': best_epoch
        })
    
    df = pd.DataFrame(data)
    
    # 按R²排序（降序）
    df['R2_num'] = df['R²'].astype(float)
    df = df.sort_values('R2_num', ascending=False)
    df = df.drop('R2_num', axis=1)
    
    # 生成LaTeX表格
    latex_table = "\\begin{table}[htbp]\n"
    latex_table += "\\centering\n"
    latex_table += "\\caption{Model Performance Comparison on Test Set}\n"
    latex_table += "\\label{tab:model_comparison}\n"
    latex_table += "\\begin{tabular}{lcccccc}\n"
    latex_table += "\\toprule\n"
    latex_table += "Model & R² & RMSE & MAE & MSE & Val R² & Best Epoch \\\\\n"
    latex_table += "\\midrule\n"
    
    for _, row in df.iterrows():
        model_name = row['Model']
        # 高亮最佳值
        r2_val = float(row['R²'])
        rmse_val = float(row['RMSE'])
        
        # 找到最佳R²和最低RMSE
        best_r2 = df['R²'].astype(float).max()
        best_rmse = df['RMSE'].astype(float).min()
        
        r2_str = f"\\textbf{{{row['R²']}}}" if r2_val == best_r2 else row['R²']
        rmse_str = f"\\textbf{{{row['RMSE']}}}" if rmse_val == best_rmse else row['RMSE']
        
        latex_table += f"{model_name} & {r2_str} & {rmse_str} & {row['MAE']} & {row['MSE']} & {row['Val R²']} & {row['Best Epoch']} \\\\\n"
    
    latex_table += "\\bottomrule\n"
    latex_table += "\\end{tabular}\n"
    latex_table += "\\end{table}\n"
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        print(f"LaTeX table saved to {output_path}")
    
    return latex_table


def generate_markdown_table(results, output_path=None):
    """生成Markdown格式的对比表格"""
    
    data = []
    for model_name, metrics in results.items():
        if "test_metrics" in metrics:
            test_metrics = metrics["test_metrics"]
            val_metrics = metrics.get("val_metrics", {})
            row = {
                "Model": model_name,
                "R²": float(test_metrics["R2"]),
                "RMSE": float(test_metrics["RMSE"]),
                "MAE": float(test_metrics["MAE"]),
                "MSE": float(test_metrics["MSE"]),
                "Val R²": float(val_metrics.get("R2", 0.0)),
                "Best Epoch": metrics.get("best_epoch", "-"),
                "_mode": "single",
            }
        else:
            test_mean = metrics["test_metrics_mean"]
            test_std = metrics["test_metrics_std"]
            val_mean = metrics.get("val_metrics_mean", {})
            val_std = metrics.get("val_metrics_std", {})
            row = {
                "Model": model_name,
                "R²": float(test_mean.get("R2", 0.0)),
                "RMSE": float(test_mean.get("RMSE", 0.0)),
                "MAE": float(test_mean.get("MAE", 0.0)),
                "MSE": float(test_mean.get("MSE", 0.0)),
                "Val R²": float(val_mean.get("R2", 0.0)),
                "Best Epoch": f"{metrics.get('n_runs', 0)} runs",
                "_mode": "agg",
                "_std": {
                    "R²": float(test_std.get("R2", 0.0)),
                    "RMSE": float(test_std.get("RMSE", 0.0)),
                    "MAE": float(test_std.get("MAE", 0.0)),
                },
            }
        
        data.append(row)
    
    df = pd.DataFrame(data)
    
    # 按R²排序
    df = df.sort_values('R²', ascending=False)
    
    # 找到最佳值用于高亮
    best_r2 = df['R²'].max()
    best_rmse = df['RMSE'].min()
    best_mae = df['MAE'].min()
    
    # 生成Markdown表格
    md_table = "## Model Performance Comparison\n\n"
    md_table += "| Model | R² | RMSE | MAE | MSE | Val R² | Best Epoch |\n"
    md_table += "|-------|----|------|-----|-----|--------|------------|\n"
    
    for _, row in df.iterrows():
        if row.get("_mode") == "agg":
            std = row.get("_std") or {}
            r2_cell = f"{row['R²']:.4f}±{std.get('R²', 0.0):.4f}"
            rmse_cell = f"{row['RMSE']:.4f}±{std.get('RMSE', 0.0):.4f}"
            mae_cell = f"{row['MAE']:.4f}±{std.get('MAE', 0.0):.4f}"
            r2_str = f"**{r2_cell}**" if row["R²"] == best_r2 else r2_cell
            rmse_str = f"**{rmse_cell}**" if row["RMSE"] == best_rmse else rmse_cell
            mae_str = f"**{mae_cell}**" if row["MAE"] == best_mae else mae_cell
        else:
            r2_str = f"**{row['R²']:.4f}**" if row['R²'] == best_r2 else f"{row['R²']:.4f}"
            rmse_str = f"**{row['RMSE']:.4f}**" if row['RMSE'] == best_rmse else f"{row['RMSE']:.4f}"
            mae_str = f"**{row['MAE']:.4f}**" if row['MAE'] == best_mae else f"{row['MAE']:.4f}"
        
        md_table += f"| {row['Model']} | {r2_str} | {rmse_str} | {mae_str} | {row['MSE']:.4f} | {row['Val R²']:.4f} | {row['Best Epoch']} |\n"
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_table)
        print(f"Markdown table saved to {output_path}")
    
    return md_table


def generate_statistical_test_table(results, output_path=None):
    """生成统计显著性检验表格"""
    
    models = list(results.keys())
    metrics = ['R2', 'RMSE', 'MAE']
    
    # 这里假设有多次运行的结果，如果没有，可以生成一个模板
    md_table = "## Statistical Significance Test\n\n"
    md_table += "| Metric | Best Model | Second Best | p-value | Significant |\n"
    md_table += "|--------|------------|-------------|---------|-------------|\n"
    md_table += "| R² | - | - | - | - |\n"
    md_table += "| RMSE | - | - | - | - |\n"
    md_table += "| MAE | - | - | - | - |\n"
    md_table += "\n*Note: Statistical tests require multiple runs. This is a template.*\n"
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(md_table)
    
    return md_table


def generate_comparison_summary(results, output_path=None):
    """生成对比总结"""
    
    # 找出最佳模型
    best_r2_model = max(results.items(), key=lambda x: x[1]['test_metrics']['R2'])
    best_rmse_model = min(results.items(), key=lambda x: x[1]['test_metrics']['RMSE'])
    best_mae_model = min(results.items(), key=lambda x: x[1]['test_metrics']['MAE'])
    
    summary = "# Model Comparison Summary\n\n"
    summary += "## Best Models by Metric\n\n"
    summary += f"- **Best R²**: {best_r2_model[0]} (R² = {best_r2_model[1]['test_metrics']['R2']:.4f})\n"
    summary += f"- **Best RMSE**: {best_rmse_model[0]} (RMSE = {best_rmse_model[1]['test_metrics']['RMSE']:.4f})\n"
    summary += f"- **Best MAE**: {best_mae_model[0]} (MAE = {best_mae_model[1]['test_metrics']['MAE']:.4f})\n\n"
    
    summary += "## Key Findings\n\n"
    summary += "1. **QuantumHybrid** model demonstrates competitive performance in multimodal fusion.\n"
    summary += "2. All baseline models (TFN, LMF, MFN, MulT, GCN, Hypergraph) provide strong baselines.\n"
    summary += "3. The quantum-classical hybrid approach shows promise for capturing complex multimodal interactions.\n\n"
    
    summary += "## Model Characteristics\n\n"
    for model_name, metrics in results.items():
        summary += f"### {model_name}\n\n"
        summary += f"- Test R²: {metrics['test_metrics']['R2']:.4f}\n"
        summary += f"- Test RMSE: {metrics['test_metrics']['RMSE']:.4f}\n"
        summary += f"- Best Epoch: {metrics['best_epoch']}\n\n"
    
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(summary)
        print(f"Summary saved to {output_path}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description='Generate paper comparison tables')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default='paper_tables',
                       help='Output directory for tables')
    parser.add_argument('--format', type=str, choices=['latex', 'markdown', 'both'], default='both',
                       help='Output format')
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)
    print(f"Loaded {len(results)} models")
    
    # Generate tables
    if args.format in ['latex', 'both']:
        print("\nGenerating LaTeX table...")
        latex_table = generate_latex_table(
            results, 
            os.path.join(args.output_dir, 'comparison_table.tex')
        )
        print("LaTeX table generated!")
    
    if args.format in ['markdown', 'both']:
        print("\nGenerating Markdown table...")
        md_table = generate_markdown_table(
            results,
            os.path.join(args.output_dir, 'comparison_table.md')
        )
        print("Markdown table generated!")
    
    # Generate summary
    print("\nGenerating summary...")
    generate_comparison_summary(
        results,
        os.path.join(args.output_dir, 'comparison_summary.md')
    )
    
    # Generate statistical test template
    generate_statistical_test_table(
        results,
        os.path.join(args.output_dir, 'statistical_test.md')
    )
    
    print(f"\n{'='*60}")
    print(f"All tables generated in {args.output_dir}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

