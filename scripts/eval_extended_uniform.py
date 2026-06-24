"""
Evaluate ExtendedUniform / TailedUniform / Uniform posteriors — C2ST vs reference.

Loads trained models from notebooks-clean/toy-2-dim-models/, evaluates at distance
bins from prior center out to 6σ extrapolation, saves figures and raw samples.
"""
import argparse
import os
import sys
import pickle
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.imports import *
from toolbox.evaluators import DistanceEvaluator


class ExtendedDistanceEvaluator(DistanceEvaluator):
    """DistanceEvaluator extended with 4σ and 6σ extrapolation bins."""

    def create_test_points(self, n_points_per_radius=50):
        distance_labels = [
            'center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0',
            '2sigma-extrap', '4sigma-extrap', '6sigma-extrap',
        ]
        radii = [
            0.0, 0.25, 0.5, 0.75, 1.0,
            2.0 * self.sigma,
            4.0 * self.sigma,
            6.0 * self.sigma,
        ]
        extrap_bins = {'2sigma-extrap', '4sigma-extrap', '6sigma-extrap'}

        test_points   = []
        distance_bins = []

        for radius, label in zip(radii, distance_labels):
            use_filter = label not in extrap_bins

            if radius == 0:
                test_points.append(self.prior_center.copy())
                distance_bins.append(label)
                continue

            points_at_radius = []
            attempts = 0
            while len(points_at_radius) < n_points_per_radius and attempts < 20:
                count = n_points_per_radius * 3 if use_filter else n_points_per_radius
                new_pts = self.sample_on_sphere(count, radius, self.dim,
                                               filter_bounds=use_filter)
                if len(new_pts) > 0:
                    points_at_radius.extend(new_pts)
                attempts += 1

            points_at_radius = points_at_radius[:n_points_per_radius]
            if len(points_at_radius) < n_points_per_radius:
                print(f'Warning: only {len(points_at_radius)}/{n_points_per_radius} '
                      f'points for {label} (r={radius:.3f})')

            for pt in points_at_radius:
                test_points.append(pt)
                distance_bins.append(label)

        return np.array(test_points), np.array(distance_bins)


def c2st(X1, X2):
    X = np.vstack([X1, X2])
    y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    return hamming_loss(y_te, LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te))


def plot_all(c2st_results, model_names, distance_bin_order, delta_scales, out_dir):
    style_map = {
        'Uniform':       {'color': '#555555', 'marker': 'o', 'ls': '-',  'lw': 2.5},
        'TailedUniform': {'color': '#e07b00', 'marker': 's', 'ls': '--', 'lw': 2.5},
        **{
            f'ExtUniform δ={d}': {'color': col, 'marker': 'D', 'ls': ':', 'lw': 2.0}
            for d, col in zip(delta_scales, ['#a8d8ea', '#5eafd1', '#2676ae', '#0d3b6e'])
        }
    }

    x_pos   = np.arange(len(distance_bin_order))
    offsets = np.linspace(-0.3, 0.3, len(model_names))

    fig, ax = plt.subplots(figsize=(13, 5))
    for i, name in enumerate(model_names):
        means, stds, xs = [], [], []
        for j, label in enumerate(distance_bin_order):
            vals = np.array(c2st_results[name][label])
            if len(vals):
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(x_pos[j] + offsets[i])
        s = style_map[name]
        ax.errorbar(xs, means, yerr=stds, fmt=s['marker'],
                    color=s['color'], label=name,
                    capsize=3, capthick=1.5, linewidth=0,
                    elinewidth=1.5, markersize=7, alpha=0.9)

    ax.axvline(4.5, color='red', linestyle='--', linewidth=2, alpha=0.85,
               label='Declared prior boundary')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Ideal (C2ST=0.5)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Center','r=0.25','r=0.5','r=0.75','r=1.0',
                        '2σ extrap','4σ extrap','6σ extrap'], fontsize=12)
    ax.set_xlabel('Distance from prior center', fontsize=14)
    ax.set_ylabel('C2ST vs Reference', fontsize=14)
    ax.set_title('ExtendedUniform vs TailedUniform vs Uniform — C2ST at all distances', fontsize=14)
    ax.legend(fontsize=9, ncol=2, loc='upper left')
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(out_dir, 'extended-uniform-c2st-all.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved {path}')


def plot_extrap(c2st_results, model_names, delta_scales, out_dir):
    extrap_bins   = ['2sigma-extrap', '4sigma-extrap', '6sigma-extrap']
    extrap_labels = ['2σ extrap', '4σ extrap', '6σ extrap']
    x_ext = np.arange(len(extrap_bins))

    fig, ax = plt.subplots(figsize=(8, 5))
    for name, color, marker in [('Uniform', '#555555', 'o'), ('TailedUniform', '#e07b00', 's')]:
        if name not in c2st_results:
            continue
        means = [np.mean(c2st_results[name][b]) for b in extrap_bins]
        stds  = [np.std(c2st_results[name][b])  for b in extrap_bins]
        ax.errorbar(x_ext, means, yerr=stds, fmt=f'-{marker}', color=color,
                    label=name, capsize=4, linewidth=2, markersize=8)

    for d_scale, col in zip(delta_scales, ['#a8d8ea', '#5eafd1', '#2676ae', '#0d3b6e']):
        name = f'ExtUniform δ={d_scale}'
        if name not in c2st_results:
            continue
        means = [np.mean(c2st_results[name][b]) for b in extrap_bins]
        stds  = [np.std(c2st_results[name][b])  for b in extrap_bins]
        ax.errorbar(x_ext, means, yerr=stds, fmt='--D', color=col,
                    label=f'ExtUniform δ={d_scale}', capsize=4, linewidth=2, markersize=8)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xticks(x_ext)
    ax.set_xticklabels(extrap_labels, fontsize=13)
    ax.set_xlabel('Extrapolation distance', fontsize=14)
    ax.set_ylabel('C2ST vs Reference', fontsize=14)
    ax.set_title('C2ST at extrapolation bins — δ sweep', fontsize=14)
    ax.legend(fontsize=10, ncol=2)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(out_dir, 'extended-uniform-c2st-extrap.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved {path}')


def plot_matched(c2st_results, distance_bin_order, out_dir):
    matched_names   = ['Uniform', 'TailedUniform', 'ExtUniform δ=0.1']
    matched_colors  = ['#555555', '#e07b00', '#2676ae']
    matched_markers = ['o', 's', 'D']
    offsets_m = [-0.15, 0.0, 0.15]
    x_pos_m   = np.arange(len(distance_bin_order))

    fig, ax = plt.subplots(figsize=(12, 5))
    for name, col, mk, off in zip(matched_names, matched_colors, matched_markers, offsets_m):
        if name not in c2st_results:
            print(f'Skipping {name} (not loaded)')
            continue
        means, stds, xs = [], [], []
        for j, label in enumerate(distance_bin_order):
            vals = np.array(c2st_results[name][label])
            if len(vals):
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(x_pos_m[j] + off)
        ax.errorbar(xs, means, yerr=stds, fmt=mk, color=col, label=name,
                    capsize=3, capthick=1.5, linewidth=0,
                    elinewidth=1.5, markersize=8, alpha=0.95)

    ax.axvline(4.5, color='red', linestyle='--', linewidth=2, alpha=0.85,
               label='Declared prior boundary')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7)
    ax.set_xticks(x_pos_m)
    ax.set_xticklabels(['Center','r=0.25','r=0.5','r=0.75','r=1.0',
                        '2σ extrap','4σ extrap','6σ extrap'], fontsize=12)
    ax.set_xlabel('Distance from prior center', fontsize=14)
    ax.set_ylabel('C2ST vs Reference', fontsize=14)
    ax.set_title('Matched reach (σ = δ = 0.1): shape matters?', fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(False)
    plt.tight_layout()
    path = os.path.join(out_dir, 'extended-uniform-c2st-matched.pdf')
    plt.savefig(path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f'Saved {path}')


def main():
    parser = argparse.ArgumentParser(description='Evaluate ExtendedUniform posteriors')
    parser.add_argument('--models_root', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'toy-2-dim-models'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'experiment-figures'))
    parser.add_argument('--n_posterior_samples', type=int, default=2000)
    parser.add_argument('--n_points_per_radius', type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Reproducibility
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    sns.set(style='whitegrid', context='paper', font_scale=1.2)

    # Task + evaluator
    param_ranges = [(-1.0, 1.0), (-1.0, 1.0)]
    delta_scales = [0.1, 0.3, 0.5, 1.0]
    task      = GaussianLinear(dim=2, prior_scale=1.0)
    simulator = task.get_simulator()
    evaluator = ExtendedDistanceEvaluator(simulator, param_ranges, task)
    print(f'Prior center: {evaluator.prior_center}')
    print(f'σ: {evaluator.sigma:.4f}  2σ={2*evaluator.sigma:.4f}  4σ={4*evaluator.sigma:.4f}  6σ={6*evaluator.sigma:.4f}')

    # Load models
    model_paths = {
        'Uniform':       os.path.join(args.models_root, 'uniform',       'posterior.pkl'),
        'TailedUniform': os.path.join(args.models_root, 'taileduniform', 'posterior.pkl'),
    }
    for d in delta_scales:
        model_paths[f'ExtUniform δ={d}'] = os.path.join(
            args.models_root, f'extended-uniform-delta-{d}', 'posterior.pkl')

    all_posteriors = {}
    for name, path in model_paths.items():
        try:
            with open(path, 'rb') as f:
                all_posteriors[name] = pickle.load(f)
            print(f'Loaded: {name}')
        except FileNotFoundError:
            print(f'NOT FOUND (skipped): {path}')

    print(f'Total models: {len(all_posteriors)}')

    # Test points
    test_points, distance_bins = evaluator.create_test_points(
        n_points_per_radius=args.n_points_per_radius)
    print(f'Test points: {len(test_points)}')

    distance_bin_order = ['center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0',
                          '2sigma-extrap', '4sigma-extrap', '6sigma-extrap']

    # Evaluate
    print('\nEvaluating posteriors...')
    results_all = evaluator.evaluate_all(all_posteriors, test_points,
                                         n_samples=args.n_posterior_samples)

    # Save raw posterior samples
    samples_path = os.path.join(args.out_dir, 'extended-uniform-posterior-samples.pkl')
    payload = {
        'results_all':    results_all,
        'test_points':    test_points,
        'distance_bins':  distance_bins,
        'distance_bin_order': distance_bin_order,
    }
    with open(samples_path, 'wb') as f:
        pickle.dump(payload, f)
    print(f'Saved raw posterior samples to {samples_path}')

    # Compute C2ST
    print('\nComputing C2ST vs reference...')
    model_names = [k for k in results_all if k not in ('test_points', 'observations', 'Reference')]
    c2st_results = {}
    for name in model_names:
        c2st_results[name] = {}
        for label in distance_bin_order:
            idx  = np.where(distance_bins == label)[0]
            vals = [c2st(results_all[name][i], results_all['Reference'][i]) for i in idx]
            c2st_results[name][label] = vals

    # Save C2ST results
    c2st_path = os.path.join(args.out_dir, 'extended-uniform-c2st-results.pkl')
    with open(c2st_path, 'wb') as f:
        pickle.dump({'c2st_results': c2st_results, 'model_names': model_names,
                     'distance_bin_order': distance_bin_order, 'delta_scales': delta_scales}, f)
    print(f'Saved C2ST results to {c2st_path}')

    # Print summary
    print(f'\nC2ST vs Reference (mean ± std)')
    print(f'{"":22s}', end='')
    for label in distance_bin_order:
        print(f'{label:20s}', end='')
    print()
    for name in model_names:
        print(f'{name:22s}', end='')
        for label in distance_bin_order:
            vals = np.array(c2st_results[name][label])
            if len(vals):
                print(f'{np.mean(vals):.3f}±{np.std(vals):.3f}      ', end='')
            else:
                print(f'{"N/A":20s}', end='')
        print()

    # Figures
    print('\nSaving figures...')
    plot_all(c2st_results, model_names, distance_bin_order, delta_scales, args.out_dir)
    plot_extrap(c2st_results, model_names, delta_scales, args.out_dir)
    plot_matched(c2st_results, distance_bin_order, args.out_dir)
    print('Done.')


if __name__ == '__main__':
    main()
