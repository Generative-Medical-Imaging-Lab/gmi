#!/usr/bin/env python3
"""
Evaluation script for all experiments in the modular configs example.
Reads final test metrics from all experiment folders and creates comprehensive box plots.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import sys

def get_experiment_names(outputs_dir="gmi_data/outputs"):
    outputs_path = Path(outputs_dir)
    if not outputs_path.exists():
        raise FileNotFoundError(f"Outputs directory not found: {outputs_dir}")
    experiment_names = []
    for item in outputs_path.iterdir():
        if item.is_dir() and (item / "final_test_metrics_per_sample.csv").exists():
            experiment_names.append(item.name)
    experiment_names.sort()
    return experiment_names

def extract_final_per_sample_metrics(experiment_name, outputs_dir="gmi_data/outputs"):
    metrics_file = Path(outputs_dir) / experiment_name / "final_test_metrics_per_sample.csv"
    if not metrics_file.exists():
        print(f"Warning: No final_test_metrics_per_sample.csv found for {experiment_name}")
        return None
    try:
        df = pd.read_csv(metrics_file)
        # Expect columns: rmse, psnr, ssim, lpips (and possibly others)
        metrics = ['rmse', 'psnr', 'ssim', 'lpips']
        data = {metric: df[metric].dropna().values for metric in metrics if metric in df.columns}
        return data
    except Exception as e:
        print(f"Error reading {metrics_file}: {e}")
        return None

def create_comprehensive_boxplot(all_data, experiment_names, output_dir, dataset_name):
    output_file = output_dir / f"{dataset_name}_experiment_comparison.png"
    summary_file = output_dir / f"{dataset_name}_experiment_summary_statistics.csv"
    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(f'{dataset_name.capitalize()} Experiment Comparison - All Metrics', fontsize=16, fontweight='bold')
    metrics = ['rmse', 'psnr', 'ssim', 'lpips']
    metric_labels = ['RMSE', 'PSNR', 'SSIM', 'LPIPS']
    metric_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        data_for_metric = []
        exp_labels = []
        for exp_name in experiment_names:
            exp_data = all_data.get(exp_name)
            if exp_data is not None and metric in exp_data:
                metric_values = exp_data[metric]
                if len(metric_values) > 0:
                    data_for_metric.append(metric_values)
                    exp_labels.append(exp_name)
        if not data_for_metric:
            ax.text(0.5, 0.5, f'No data for {label}', ha='center', va='center', transform=ax.transAxes)
            continue
        bp = ax.boxplot(data_for_metric, patch_artist=True, tick_labels=exp_labels)
        for patch in bp['boxes']:
            patch.set_facecolor(metric_colors[i])
            patch.set_alpha(0.7)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        if data_for_metric:
            means = [np.mean(data) for data in data_for_metric]
            ax.plot(range(1, len(means) + 1), means, 'ko', markersize=4, label='Mean')
            ax.legend()
    axes[-1].set_xlabel('Experiment Name', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{dataset_name.capitalize()} comparison plot saved as: {output_file}")
    save_summary_statistics(all_data, experiment_names, summary_file)
    plt.close(fig)

def create_bar_chart_means(all_data, experiment_names, output_dir, dataset_name):
    output_file = output_dir / f"{dataset_name}_experiment_bar_means.png"
    metrics = ['rmse', 'psnr', 'ssim', 'lpips']
    metric_labels = ['RMSE', 'PSNR', 'SSIM', 'LPIPS']
    metric_colors = ['#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
    means = {metric: [] for metric in metrics}
    for exp_name in experiment_names:
        exp_data = all_data.get(exp_name)
        for i, metric in enumerate(metrics):
            if exp_data is not None and metric in exp_data:
                values = exp_data[metric]
                means[metric].append(np.mean(values) if len(values) > 0 else np.nan)
            else:
                means[metric].append(np.nan)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    fig.suptitle(f'{dataset_name.capitalize()} Experiment Means - All Metrics', fontsize=16, fontweight='bold')
    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        ax = axes[i]
        ax.bar(experiment_names, means[metric], color=metric_colors[i], alpha=0.8)
        ax.set_ylabel(label, fontsize=12, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        ax.tick_params(axis='x', rotation=45, labelsize=10)
        ax.tick_params(axis='y', labelsize=10)
        for j, v in enumerate(means[metric]):
            if not np.isnan(v):
                ax.text(j, v, f'{v:.3g}', ha='center', va='bottom', fontsize=8)
    axes[-1].set_xlabel('Experiment Name', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"{dataset_name.capitalize()} bar chart means saved as: {output_file}")
    plt.close(fig)

def save_summary_statistics(all_data, experiment_names, output_file):
    summary_data = []
    for exp_name in experiment_names:
        exp_data = all_data.get(exp_name)
        if exp_data is None:
            continue
        metrics = ['rmse', 'psnr', 'ssim', 'lpips']
        row = {'experiment': exp_name}
        for metric in metrics:
            if metric in exp_data:
                values = exp_data[metric]
                if len(values) > 0:
                    row[f'{metric}_mean'] = np.mean(values)
                    row[f'{metric}_std'] = np.std(values)
                    row[f'{metric}_min'] = np.min(values)
                    row[f'{metric}_max'] = np.max(values)
                    row[f'{metric}_count'] = len(values)
                else:
                    row[f'{metric}_mean'] = np.nan
                    row[f'{metric}_std'] = np.nan
                    row[f'{metric}_min'] = np.nan
                    row[f'{metric}_max'] = np.nan
                    row[f'{metric}_count'] = 0
            else:
                row[f'{metric}_mean'] = np.nan
                row[f'{metric}_std'] = np.nan
                row[f'{metric}_min'] = np.nan
                row[f'{metric}_max'] = np.nan
                row[f'{metric}_count'] = 0
        summary_data.append(row)
    df = pd.DataFrame(summary_data)
    df.to_csv(output_file, index=False)
    print(f"Summary statistics saved as: {output_file}")

def main():
    print("Starting comprehensive experiment evaluation...")
    script_dir = Path(__file__).parent.resolve()
    outputs_dir = script_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    experiment_names = get_experiment_names()
    print(f"Found {len(experiment_names)} experiments:")
    for name in experiment_names:
        print(f"  - {name}")
    all_data = {}
    for exp_name in experiment_names:
        print(f"Processing {exp_name}...")
        data = extract_final_per_sample_metrics(exp_name)
        all_data[exp_name] = data
        if data:
            print(f"  - Extracted per-sample final test metrics")
        else:
            print(f"  - No data found")
    # Group by dataset
    datasets = ['mnist', 'chestmnist', 'bloodmnist']
    for dataset in datasets:
        dataset_experiments = [name for name in experiment_names if name.startswith(dataset)]
        if not dataset_experiments:
            print(f"No experiments found for dataset: {dataset}")
            continue
        print(f"\nCreating plot for {dataset}...")
        create_comprehensive_boxplot(all_data, dataset_experiments, outputs_dir, dataset)
        create_bar_chart_means(all_data, dataset_experiments, outputs_dir, dataset)
    print("\nEvaluation complete!")

if __name__ == "__main__":
    main() 