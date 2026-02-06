"""
Compare performance of different models
"""

import json
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def load_results(results_dir):
    """Load results from JSON file"""
    results_path = os.path.join(results_dir, 'all_results.json')
    if not os.path.exists(results_path):
        raise FileNotFoundError(f"Results file not found: {results_path}")
    
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return results


def create_comparison_table(results):
    """Create a comparison table"""
    data = []
    for model_name, metrics in results.items():
        test_metrics = metrics['test_metrics']
        data.append({
            'Model': model_name,
            'MSE': test_metrics['MSE'],
            'MAE': test_metrics['MAE'],
            'RMSE': test_metrics['RMSE'],
            'R2': test_metrics['R2'],
            'MAPE': test_metrics['MAPE'],
            'Best Val Loss': metrics['best_val_loss'],
            'Best Epoch': metrics['best_epoch']
        })
    
    df = pd.DataFrame(data)
    return df


def plot_comparison(results, save_path=None):
    """Plot comparison of different models"""
    models = list(results.keys())
    metrics_to_plot = ['MSE', 'MAE', 'RMSE', 'R2']
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        values = [results[model]['test_metrics'][metric] for model in models]
        
        axes[idx].bar(models, values, alpha=0.7)
        axes[idx].set_title(f'Test {metric} Comparison')
        axes[idx].set_ylabel(metric)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_radar_chart(results, save_path=None):
    """Create a radar chart comparing models"""
    models = list(results.keys())
    metrics = ['R2', 'RMSE', 'MAE', 'MSE']
    
    # Normalize metrics (R2 is higher better, others are lower better)
    normalized_data = {}
    for model in models:
        test_metrics = results[model]['test_metrics']
        normalized = {}
        
        # Normalize R2 (0-1 scale, assume max is 1.0)
        normalized['R2'] = test_metrics['R2']
        
        # Normalize others (inverse, so higher is better)
        max_rmse = max([results[m]['test_metrics']['RMSE'] for m in models])
        max_mae = max([results[m]['test_metrics']['MAE'] for m in models])
        max_mse = max([results[m]['test_metrics']['MSE'] for m in models])
        
        normalized['RMSE'] = 1 - (test_metrics['RMSE'] / max_rmse)
        normalized['MAE'] = 1 - (test_metrics['MAE'] / max_mae)
        normalized['MSE'] = 1 - (test_metrics['MSE'] / max_mse)
        
        normalized_data[model] = normalized
    
    # Create radar chart
    angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
    angles += angles[:1]  # Complete the circle
    
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(models)))
    
    for idx, model in enumerate(models):
        values = [normalized_data[model][m] for m in metrics]
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[idx])
        ax.fill(angles, values, alpha=0.25, color=colors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1)
    ax.set_title('Model Comparison (Normalized Metrics)', size=16, pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Compare model performance')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Directory containing results')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for comparison plots')
    args = parser.parse_args()
    
    output_dir = args.output_dir or args.results_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    print("Loading results...")
    results = load_results(args.results_dir)
    
    # Create comparison table
    print("Creating comparison table...")
    df = create_comparison_table(results)
    print("\n" + "="*80)
    print("Model Comparison Table")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    
    # Save table
    table_path = os.path.join(output_dir, 'comparison_table.csv')
    df.to_csv(table_path, index=False)
    print(f"\nComparison table saved to {table_path}")
    
    # Create plots
    print("\nCreating comparison plots...")
    plot_comparison(results, os.path.join(output_dir, 'comparison_bar.png'))
    plot_radar_chart(results, os.path.join(output_dir, 'comparison_radar.png'))
    print(f"Plots saved to {output_dir}")
    
    # Find best model
    print("\n" + "="*80)
    print("Best Models by Metric:")
    print("="*80)
    best_r2 = df.loc[df['R2'].idxmax(), 'Model']
    best_rmse = df.loc[df['RMSE'].idxmin(), 'Model']
    best_mae = df.loc[df['MAE'].idxmin(), 'Model']
    
    print(f"Best R2: {best_r2} (R2 = {df.loc[df['R2'].idxmax(), 'R2']:.4f})")
    print(f"Best RMSE: {best_rmse} (RMSE = {df.loc[df['RMSE'].idxmin(), 'RMSE']:.4f})")
    print(f"Best MAE: {best_mae} (MAE = {df.loc[df['MAE'].idxmin(), 'MAE']:.4f})")
    print("="*80)


if __name__ == '__main__':
    main()







