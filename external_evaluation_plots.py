"""Aggregate a complete k sweep and plot external metrics for both models."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {'robust_ssc': 'Robust SSC', 'robust_lrsc': 'Robust LRSC'}
METRICS = {'ARI': 'Adjusted Rand index', 'NMI': 'Normalized mutual information', 'Purity': 'Cluster purity'}


def create_plots(root, k_min=3, k_max=10):
    """Validate a complete sweep before exporting comparison tables and figures."""
    rows = []
    for method in METHODS:
        for k in range(k_min, k_max + 1):
            folder = root / 'step_10' / method / f'k_{k}'
            metrics = pd.read_csv(folder / 'external_metrics.csv')
            if len(metrics) != 1 or not set(METRICS).issubset(metrics.columns):
                raise ValueError(f'Invalid metric table: {folder}')
            values = metrics.iloc[0][list(METRICS)].astype(float)
            if not np.isfinite(values).all():
                raise ValueError(f'Nonfinite external metric: {folder}')
            labels = pd.read_csv(root / 'step_6' / method / f'k_{k}' / 'labels.csv')
            if labels['Cluster'].nunique() != k:
                raise ValueError(f'Expected {k} clusters for {method}')
            rows.append(dict(Method=method, k=k, Samples=len(labels), **values.to_dict()))
    data = pd.DataFrame(rows)
    output = root / 'external_evaluation_graphs'
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'external_metrics_by_k.csv', index=False)
    plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})

    def draw(ax, metric):
        for (method, label), color, marker in zip(METHODS.items(), ['#1565a8', '#d45b27'], ['o', 's']):
            frame = data[data.Method == method]
            ax.plot(frame.k, frame[metric], color=color, marker=marker, linewidth=2, label=label)
        ax.set(xlabel='Number of clusters (k)', ylabel=METRICS[metric], xticks=range(k_min, k_max + 1))
        if metric != 'ARI':
            ax.set_ylim(0, 1.03)
        else:
            ax.set_ylim(min(-0.05, data.ARI.min() - 0.05), 1.03)
        ax.grid(alpha=0.2)
        ax.legend(frameon=False)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
        draw(ax, metric)
        ax.set_title(f'{METRICS[metric]} across cluster counts')
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{metric.lower()}_by_k.{ext}', dpi=250)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), layout='constrained')
    for ax, metric in zip(axes, METRICS):
        draw(ax, metric)
    fig.suptitle('External evaluation · same data and fixed model parameters', fontsize=16)
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'external_evaluation_comparison.{ext}', dpi=250)
    plt.close(fig)
    print(f'Saved {len(data)} model/k evaluations and graphs to {output}')
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--k-min', type=int, default=3)
    parser.add_argument('--k-max', type=int, default=10)
    args = parser.parse_args()
    if not 2 <= args.k_min <= args.k_max:
        parser.error('Require 2 <= k-min <= k-max')
    create_plots(args.root, args.k_min, args.k_max)
