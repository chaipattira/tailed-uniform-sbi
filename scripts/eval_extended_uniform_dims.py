"""
Evaluate posteriors for inference-dim-waste experiment.
Loops over dims 4, 8, 16; evaluates C2ST vs reference at distance bins.
Missing posteriors are skipped with a warning.
"""
import argparse
import os
import sys
import pickle
import random

import numpy as np
import torch
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.imports import *
from toolbox.evaluators import DistanceEvaluator

DIMS          = [4, 8, 16]
DELTA_SCALES  = [0.1, 0.3]
DISTANCE_BIN_ORDER = ['center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0', '2sigma-extrap']


def c2st(X1, X2):
    X  = np.vstack([X1, X2])
    y  = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    return hamming_loss(y_te, LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_root', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'inference-dim-models'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'inference-dim-figures'))
    parser.add_argument('--n_posterior_samples', type=int, default=2000)
    parser.add_argument('--n_points_per_radius',  type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    sns.set(style='whitegrid', context='paper', font_scale=1.2)

    all_results = {}

    for dim in DIMS:
        print(f'\n{"="*60}')
        print(f'dim={dim}')
        print(f'{"="*60}')

        param_ranges = [(-1.0, 1.0)] * dim
        task         = GaussianLinear(dim=dim, prior_scale=1.0)
        simulator    = task.get_simulator()
        evaluator    = DistanceEvaluator(simulator, param_ranges, task)
        print(f'σ={evaluator.sigma:.4f}  2σ={2*evaluator.sigma:.4f}')

        # Build model path dict
        model_paths = {
            'Uniform':       os.path.join(args.models_root, f'dim{dim}', 'uniform',       'posterior.pkl'),
            'TailedUniform': os.path.join(args.models_root, f'dim{dim}', 'taileduniform', 'posterior.pkl'),
        }
        for d in DELTA_SCALES:
            model_paths[f'ExtUniform δ={d}'] = os.path.join(
                args.models_root, f'dim{dim}', f'extended-uniform-delta-{d}', 'posterior.pkl')

        posteriors = {}
        for name, path in model_paths.items():
            try:
                with open(path, 'rb') as f:
                    posteriors[name] = pickle.load(f)
                print(f'Loaded: {name}')
            except FileNotFoundError:
                print(f'NOT FOUND (skipped): {path}')

        if not posteriors:
            print(f'No posteriors found for dim={dim}, skipping.')
            continue

        test_points, distance_bins = evaluator.create_test_points(
            n_points_per_radius=args.n_points_per_radius)
        print(f'Test points: {len(test_points)}')

        results_all = evaluator.evaluate_all(
            posteriors, test_points, n_samples=args.n_posterior_samples)

        model_names = [k for k in results_all if k not in ('test_points', 'observations', 'Reference')]
        c2st_results = {}
        for name in model_names:
            c2st_results[name] = {}
            for label in DISTANCE_BIN_ORDER:
                idx  = np.where(distance_bins == label)[0]
                vals = [c2st(results_all[name][i], results_all['Reference'][i]) for i in idx]
                c2st_results[name][label] = vals

        all_results[dim] = {
            'c2st_results':      c2st_results,
            'model_names':       model_names,
            'distance_bin_order': DISTANCE_BIN_ORDER,
        }

        # Per-dim summary
        print(f'\nC2ST vs Reference — dim={dim}')
        print(f'{"":22s}', end='')
        for label in DISTANCE_BIN_ORDER:
            print(f'{label:20s}', end='')
        print()
        for name in model_names:
            print(f'{name:22s}', end='')
            for label in DISTANCE_BIN_ORDER:
                vals = np.array(c2st_results[name][label])
                if len(vals):
                    print(f'{np.mean(vals):.3f}±{np.std(vals):.3f}      ', end='')
                else:
                    print(f'{"N/A":20s}', end='')
            print()

    # Save combined results
    out_path = os.path.join(args.out_dir, 'inference-dims-c2st-results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved combined results to {out_path}')


if __name__ == '__main__':
    main()
