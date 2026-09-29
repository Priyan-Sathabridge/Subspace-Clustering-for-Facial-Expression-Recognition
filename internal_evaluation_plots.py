"""Compare the existing internal validation metrics across models and k."""
import argparse
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

METHODS = {'robust_ssc': 'Robust SSC', 'robust_lrsc': 'Robust LRSC'}
METRICS = {'Silhouette': 'Higher is better', 'Affinity Silhouette': 'Higher is better',
           'Affinity Ratio': 'Higher is better', 'Normalized Cut': 'Lower is better',
           'DBI': 'Lower is better', 'CHI': 'Higher is better', 'Eigengap': 'Larger gap'}


def create_plots(root, k_min=3, k_max=10):
    """Validate a complete sweep before exporting comparison tables and figures."""
    frames = []
    for method in METHODS:
        frame = pd.read_csv(root / 'step_8' / method / 'evaluation_results.csv')
        if not {'k', *METRICS}.issubset(frame.columns):
            raise ValueError(f'Missing internal metrics for {method}')
        if sorted(frame.k.tolist()) != list(range(k_min, k_max + 1)):
            raise ValueError(f'Incomplete or duplicate k values for {method}')
        if not np.isfinite(frame[list(METRICS)].to_numpy(dtype=float)).all():
            raise ValueError(f'Nonfinite internal metric for {method}; inspect evaluation results')
        counts = []
        for k in frame.k:
            labels = pd.read_csv(root / 'step_6' / method / f'k_{k}' / 'labels.csv')
            if labels.Cluster.nunique() != k:
                raise ValueError(f'Expected {k} clusters for {method}')
            counts.append(len(labels))
        frame.insert(0, 'Method', method)
        frame['Samples'] = counts
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    output = root / 'internal_evaluation_graphs'
    output.mkdir(parents=True, exist_ok=True)
    data.to_csv(output / 'internal_metrics_by_k.csv', index=False)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})

    def draw(ax, metric):
        for (method, label), color, marker in zip(METHODS.items(), ['#1565a8', '#d45b27'], ['o', 's']):
            frame = data[data.Method == method].sort_values('k')
            ax.plot(frame.k, frame[metric], color=color, marker=marker, linewidth=2, label=label)
        ax.set(xlabel='Number of clusters (k)', ylabel=metric,
               title=f'{metric} · {METRICS[metric]}', xticks=range(k_min, k_max + 1))
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, fontsize=9)

    for metric in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.5), layout='constrained')
        draw(ax, metric)
        for ext in ['png', 'pdf']:
            fig.savefig(output / f'{metric.lower().replace(" ", "_")}_by_k.{ext}', dpi=250)
        plt.close(fig)
    fig, axes = plt.subplots(3, 3, figsize=(16, 12), layout='constrained')
    for ax, metric in zip(axes.flat, METRICS):
        draw(ax, metric)
    for ax in list(axes.flat)[len(METRICS):]:
        ax.axis('off')
    fig.suptitle('Internal validation · same data and fixed model parameters', fontsize=17)
    for ext in ['png', 'pdf']:
        fig.savefig(output / f'internal_evaluation_comparison.{ext}', dpi=200)
    plt.close(fig)
    print(f'Saved {len(data)} model/k evaluations and seven metric comparisons to {output}')
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
