"""
Evaluate Exp 3 (sigma/tail-width sensitivity): C2ST vs reference at distance bins.
Loads posteriors from results/exp3-sigma/, caches samples, one plot per dim.
"""
import os, sys, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sbibm.tasks.gaussian_linear.task import GaussianLinear
from sbibm.tasks.gaussian_linear_uniform.task import GaussianLinearUniform
from toolbox.evaluators import DistanceEvaluator, c2st

DIMS             = [4, 8]
SIGMAS           = [0.05, 0.1, 0.2, 0.4]
TAILED_PROPOSALS = ['gaussiantailed', 'lineartailed', 'exponentialtailed', 'uniformtailed']
DIST_BINS        = ['center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0', '2sigma-extrap']

PROPOSAL_STYLE = {
    'uniform':           {'color': '#4477AA', 'marker': 'o'},
    'gaussiantailed':    {'color': '#EE6677', 'marker': 's'},
    'lineartailed':      {'color': '#228833', 'marker': '^'},
    'exponentialtailed': {'color': '#CCBB44', 'marker': 'D'},
    'uniformtailed':     {'color': '#AA3377', 'marker': 'v'},
}
PROPOSAL_LABEL = {
    'uniform':           'Uniform',
    'gaussiantailed':    'Gaussian-tailed',
    'lineartailed':      'Linear-tailed',
    'exponentialtailed': 'Exp-tailed',
    'uniformtailed':     'Uniform-tailed',
}
SIGMA_LINESTYLE = {0.05: ':', 0.1: '--', 0.2: '-', 0.4: '-.'}

RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 10,
    'axes.titleweight': 'bold',
    'axes.labelsize': 9,
    'legend.fontsize': 7.5,
    'legend.frameon': False,
    'legend.borderpad': 0.4,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'lines.linewidth': 1.4,
    'lines.markersize': 4,
    'xtick.major.size': 3,
    'ytick.major.size': 3,
}


def load_posterior(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def evaluate_cell(dim, label, models_path, cache_path_base, evaluator,
                  test_points, distance_bins, n_samples):
    cache_dir  = os.path.join(cache_path_base, label)
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, 'posterior_samples.pkl')

    if os.path.exists(cache_path):
        print(f'  [{label}] Loading cached samples')
        with open(cache_path, 'rb') as f:
            results = pickle.load(f)
    else:
        pkl = os.path.join(models_path, 'posterior.pkl')
        if not os.path.exists(pkl):
            print(f'  WARNING: missing {pkl}')
            return None
        posterior = load_posterior(pkl)
        samples = []
        ref_post_list = []
        import torch
        for theta in test_points:
            x_obs = evaluator.simulator(torch.tensor(theta, dtype=torch.float32))
            ref_post_list.append(evaluator.get_reference_samples(x_obs, n_samples))
            samples.append(posterior.sample((n_samples,), x_obs).cpu().numpy())
        results = {'samples': samples, 'reference': ref_post_list,
                   'distance_bins': distance_bins}
        with open(cache_path, 'wb') as f:
            pickle.dump(results, f)

    c2st_result = {}
    for bin_name in DIST_BINS:
        idxs = np.where(results['distance_bins'] == bin_name)[0]
        vals = [c2st(results['samples'][i], results['reference'][i]) for i in idxs]
        c2st_result[bin_name] = vals
    return c2st_result


def plot_dim(dim, dim_c2st, out_path):
    plt.rcParams.update(RC_PARAMS)
    x_pos      = np.arange(len(DIST_BINS))
    bin_labels = ['Center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0', '2σ Extrap']

    ncols = 2
    nrows = (len(SIGMAS) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.75, nrows * 2.6), sharey=True)
    axes_flat = np.array(axes).flatten()

    all_proposals_ordered = ['uniform'] + list(TAILED_PROPOSALS)
    offsets    = np.linspace(-0.2, 0.2, len(all_proposals_ordered))
    boundaries = np.arange(len(DIST_BINS) + 1) - 0.5
    uniform_data = dim_c2st.get('uniform', {}).get(None, {})

    for ax_i, sigma in enumerate(SIGMAS):
        ax = axes_flat[ax_i]

        for i, pname in enumerate(all_proposals_ordered):
            if pname == 'uniform':
                data = uniform_data
                color, marker, label = '#4477AA', 'o', 'Uniform'
            else:
                data = dim_c2st.get(sigma, {}).get(pname, {})
                color  = PROPOSAL_STYLE[pname]['color']
                marker = PROPOSAL_STYLE[pname]['marker']
                label  = PROPOSAL_LABEL[pname]
            if not data:
                continue
            means = np.array([np.mean(data.get(b, [np.nan])) for b in DIST_BINS])
            errs  = np.array([np.std(data.get(b,  [0]))       for b in DIST_BINS])
            xo = x_pos + offsets[i]
            ax.errorbar(xo, means, yerr=errs,
                        fmt=marker, color=color,
                        capsize=2, markersize=4, linewidth=0, elinewidth=1.0,
                        label=label, alpha=0.9, zorder=3)

        ax.axvline(len(DIST_BINS) - 1.5, color='#CC3311', linestyle='--',
                   linewidth=1.0, alpha=0.8, zorder=0)
        ax.axhline(0.5, color='#888888', linestyle=':', linewidth=1.0)
        ax.set_xticks(boundaries)
        ax.set_xticklabels([])
        ax.tick_params(axis='x', which='major', length=3)
        ax.set_xticks(x_pos, minor=True)
        ax.set_xticklabels(bin_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
        ax.tick_params(axis='x', which='minor', length=0)
        ax.set_xlim(-0.5, len(DIST_BINS) - 0.5)
        ax.set_title(f'σ = {sigma}')
        ax.yaxis.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)
        if ax_i % ncols == 0:
            ax.set_ylabel('C2ST vs Reference')
        if ax_i == 0:
            ax.legend(ncol=1, loc='lower left', handlelength=1.5)

    for j in range(len(SIGMAS), len(axes_flat)):
        axes_flat[j].set_visible(False)

    for ax_i in range((nrows - 1) * ncols, nrows * ncols):
        if ax_i < len(SIGMAS):
            axes_flat[ax_i].set_xlabel('Distance from prior centre')

    fig.suptitle(f'C2ST vs distance — sigma sensitivity (D={dim})', fontsize=11)
    plt.tight_layout()
    base = os.path.splitext(out_path)[0]
    fig.savefig(f'{base}.pdf', bbox_inches='tight')
    fig.savefig(f'{base}.png', dpi=300, bbox_inches='tight')
    print(f'Saved {base}.pdf / .png')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='gaussianlinear',
                        choices=['gaussianlinear', 'gaussianlinearuniform'])
    parser.add_argument('--models_root', default=None)
    parser.add_argument('--out_root',    default=None)
    parser.add_argument('--n_samples',   type=int, default=2000)
    parser.add_argument('--n_per_radius', type=int, default=40)
    args = parser.parse_args()
    if args.models_root is None:
        args.models_root = os.path.join(ROOT, 'results', args.task_name, 'exp3-sigma')
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', args.task_name, 'exp3-sigma')

    results_path = os.path.join(args.out_root, 'c2st_results.pkl')
    if os.path.exists(results_path):
        print(f'Loading saved C2ST results from {results_path}')
        with open(results_path, 'rb') as f:
            all_c2st = pickle.load(f)
    else:
        cache_root = os.path.join(args.out_root, 'eval-cache')
        os.makedirs(cache_root, exist_ok=True)
        all_c2st = {}
        for dim in DIMS:
            param_ranges = [(-1.0, 1.0)] * dim
            if args.task_name == 'gaussianlinear':
                task = GaussianLinear(dim=dim, prior_scale=1.0)
            else:
                task = GaussianLinearUniform(dim=dim)
            simulator    = task.get_simulator()
            evaluator    = DistanceEvaluator(simulator, param_ranges, task)
            test_points, distance_bins = evaluator.create_test_points(
                n_points_per_radius=args.n_per_radius
            )
            dim_cache = os.path.join(cache_root, f'dim{dim}')
            all_c2st[dim] = {}

            print(f'\n=== dim={dim}  proposal=uniform ===')
            r = evaluate_cell(dim, 'uniform',
                              os.path.join(args.models_root, f'dim{dim}', 'uniform'),
                              dim_cache, evaluator, test_points, distance_bins, args.n_samples)
            if r:
                all_c2st[dim]['uniform'] = {None: r}

            for sigma in SIGMAS:
                all_c2st[dim][sigma] = {}
                for pname in TAILED_PROPOSALS:
                    print(f'\n=== dim={dim}  sigma={sigma}  proposal={pname} ===')
                    mpath = os.path.join(args.models_root, f'dim{dim}', f'sigma{sigma}', pname)
                    r = evaluate_cell(dim, f'sigma{sigma}_{pname}',
                                      mpath, dim_cache, evaluator,
                                      test_points, distance_bins, args.n_samples)
                    if r:
                        all_c2st[dim][sigma][pname] = r

        with open(results_path, 'wb') as f:
            pickle.dump(all_c2st, f)
        print(f'\nSaved C2ST results to {results_path}')

    for dim in DIMS:
        plot_dim(dim, all_c2st[dim],
                 os.path.join(args.out_root, f'plot_c2st_vs_sigma_dim{dim}.pdf'))


if __name__ == '__main__':
    main()
