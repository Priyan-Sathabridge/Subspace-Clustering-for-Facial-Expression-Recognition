"""Validate and launch an isolated final run using the existing PCA dataset."""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime

DEFAULT_PROJECT = Path('/Users/priyansathabridge/Desktop/Presentation_outputs/sc_dashboard/sc_dashboard')
DEFAULT_DATA = Path('/Users/priyansathabridge/Desktop/Labelled_Expression_Data')


def read_csv(path):
    with path.open(newline='', encoding='utf-8-sig') as handle:
        return list(csv.DictReader(handle))


def validate(project, source):
    for filename in ('pipeline.py', 'pipeline_config.py'):
        if not (project / filename).is_file():
            raise ValueError(f'Missing dashboard file: {project / filename}')
    tables = {}
    for filename in ('ground_truth.csv', 'step_2/processed_features.csv', 'step_3/pca_features.csv'):
        rows = read_csv(source / filename)
        if not rows or 'Image' not in rows[0]:
            raise ValueError(f'{filename}: expected nonempty data with an Image column')
        ids = [r['Image'] for r in rows]
        if len(ids) != len(set(ids)) or any(not x or Path(x).name != x for x in ids):
            raise ValueError(f'{filename}: image identifiers must be unique, nonempty filenames')
        tables[filename] = rows
    pca = tables['step_3/pca_features.csv']
    ids = [r['Image'] for r in pca]
    for filename, rows in tables.items():
        if [r['Image'] for r in rows] != ids:
            raise ValueError(f'{filename}: image order differs from PCA; align before running')
    for filename in ('step_2/processed_features.csv', 'step_3/pca_features.csv'):
        for row in tables[filename]:
            for column, value in row.items():
                if column != 'Image' and not math.isfinite(float(value)):
                    raise ValueError(f'{filename}: nonfinite value in {column}')
    images = {}
    counts = {}
    for row in tables['ground_truth.csv']:
        label = row.get('Label', '')
        if not label:
            raise ValueError('Ground truth requires a nonempty Label for every image')
        path = Path(row.get('Path', ''))
        if not path.is_absolute():
            path = source / path
        if not path.is_file():
            raise ValueError(f'Missing source image: {path}')
        images[row['Image']] = path
        counts[label] = counts.get(label, 0) + 1
    spec = importlib.util.spec_from_file_location('final_pipeline_config', project / 'pipeline_config.py')
    config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config)
    if len(ids) <= config.N_CLUSTERS:
        raise ValueError('Configured cluster count must be smaller than the sample count')
    print(f'Validated {len(ids)} aligned samples; {len(pca[0]) - 1} PCA components.', flush=True)
    print(f'Expression counts: {counts}', flush=True)
    print(f'Final methods: {config.FINAL_METHODS}; k={config.N_CLUSTERS}', flush=True)
    print(f'SSC: lambda={config.ROBUST_SSC_LAMBDA}, gamma={config.ROBUST_SSC_GAMMA}', flush=True)
    print(f'LRSC: lambda={config.ROBUST_LRSC_LAMBDA}, gamma={config.ROBUST_LRSC_GAMMA}', flush=True)
    return images


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=DEFAULT_PROJECT)
    parser.add_argument('--source', type=Path, default=DEFAULT_DATA)
    parser.add_argument('--output-parent', type=Path, help='Default: SOURCE/final_results')
    parser.add_argument('--run', action='store_true', help='Execute stages 4–10; otherwise validate only')
    parser.add_argument('--sweep', action='store_true', help='Compare external metrics across k')
    parser.add_argument('--k-min', type=int, default=3)
    parser.add_argument('--k-max', type=int, default=10)
    args = parser.parse_args()
    project, source = args.project.resolve(), args.source.resolve()
    images = validate(project, source)
    if args.sweep:
        if not 2 <= args.k_min <= args.k_max < len(images):
            parser.error('Require 2 <= k-min <= k-max < sample count')
        print(f'External evaluation sweep: k={args.k_min} through {args.k_max}', flush=True)
    if not args.run:
        print('Input checks passed. Add --run to generate final results.')
        return 0
    # A unique directory ensures an earlier experiment is never reused or overwritten.
    parent = (args.output_parent or source / ('external_evaluation_sweeps' if args.sweep else 'final_results')).resolve()
    run = parent / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    run.mkdir(parents=True, exist_ok=False)
    env = os.environ.copy()
    env.update(SC_DATA_ROOT=str(run), SC_IMAGE_FOLDER=str(run / 'all_images'),
               SC_GROUND_TRUTH_CSV=str(run / 'ground_truth.csv'),
               MPLBACKEND='Agg', PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1',
               MPLCONFIGDIR=str(run / '.matplotlib'), NUMBA_CACHE_DIR=str(run / '.numba'))
    # Fail early on missing dependencies, before either model is fitted.
    modules = 'numpy,pandas,scipy,sklearn,matplotlib' if args.sweep else 'numpy,pandas,scipy,sklearn,matplotlib,plotly,umap,PIL,statsmodels'
    subprocess.run([sys.executable, '-c', f'import {modules}'], env=env, check=True)
    for directory in ('step_2', 'step_3'):
        shutil.copytree(source / directory, run / directory)
    shutil.copy2(source / 'ground_truth.csv', run / 'ground_truth.csv')
    (run / 'all_images').mkdir()
    for name, image in images.items():
        shutil.copy2(image, run / 'all_images' / name)
    # Snapshot the exact code and settings used so subsequent edits cannot change this run.
    snapshot = run / 'dashboard_code'
    shutil.copytree(project, snapshot, ignore=shutil.ignore_patterns(
        '__pycache__', '.DS_Store', '.git', '.venv', 'data', 'results', 'final_results'))
    hashes = {}
    for relative in ('ground_truth.csv', 'step_2/processed_features.csv', 'step_3/pca_features.csv'):
        hashes[relative] = hashlib.sha256((run / relative).read_bytes()).hexdigest()
    (run / 'input_manifest.json').write_text(json.dumps({
        'source': str(source), 'python': sys.executable, 'python_version': sys.version,
        'start_stage': 4, 'k_values': list(range(args.k_min, args.k_max + 1)) if args.sweep else None, 'methods': 'both', 'samples': len(images), 'sha256': hashes,
        'note': 'Existing processed features and PCA reused; stages 1–3 are not recomputed.',
    }, indent=2))
    with (run / 'environment.txt').open('w') as handle:
        subprocess.run([sys.executable, '-m', 'pip', 'freeze'], stdout=handle, check=True)
    print(f'Final results folder: {run}', flush=True)
    runner = [sys.executable, str(snapshot / 'pipeline.py')]
    if args.sweep:
        env['SC_K_SWEEP'] = '1'
        commands = [(runner + ['--only', '4,5', '--method', 'both'], env.copy())]
        for k in range(args.k_min, args.k_max + 1):
            k_env = dict(env, SC_N_CLUSTERS=str(k))
            commands.append((runner + ['--only', '6,10', '--method', 'both'], k_env))
        commands.append(([sys.executable, str(snapshot / 'external_evaluation_plots.py'),
                          '--root', str(run), '--k-min', str(args.k_min),
                          '--k-max', str(args.k_max)], env.copy()))
    else:
        env.pop('SC_K_SWEEP', None)
        commands = [(runner + ['--from', '4', '--method', 'both'], env)]
    code = 0
    with (run / 'run.log').open('w') as log:
        for command, command_env in commands:
            process = subprocess.Popen(command, cwd=snapshot, env=command_env, stdout=subprocess.PIPE,
                                       stderr=subprocess.STDOUT, text=True, bufsize=1)
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            code = process.wait()
            if code:
                break
    print(f'Run {"completed" if code == 0 else "failed"}. Outputs and logs: {run}')
    return code


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(f'Cannot start/complete final run: {exc}', file=sys.stderr)
        raise SystemExit(1)
