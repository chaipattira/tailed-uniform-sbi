"""
2D spatial analysis for Exp 1 D=2 models:
  1. 10x10 C2ST heatmap across [-1,1]^2 for each proposal vs reference
  2. Corner plots at (±0.8, ±0.8) with all proposals + reference overlaid
"""
import os, sys, pickle, argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sbibm.tasks.gaussian_linear.task import GaussianLinear
from sbibm.tasks.gaussian_linear_uniform.task import GaussianLinearUniform
from toolbox.evaluators import c2st

PROPOSALS = ['uniform', 'gaussiantailed', 'lineartailed', 'exponentialtailed', 'uniformtailed']
CORNERS   = {
    'ne': (0.8,  0.8),
    'nw': (-0.8, 0.8),
    'se': (0.8, -0.8),
    'sw': (-0.8,-0.8),
}
PROPOSAL_STYLE = {
    'uniform':           {'color': '#4477AA', 'marker': 'o'},
    'gaussiantailed':    {'color': '#EE6677', 'marker': 's'},
    'lineartailed':      {'color': '#228833', 'marker': '^'},
    'exponentialtailed': {'color': '#CCBB44', 'marker': 'D'},
    'uniformtailed':     {'color': '#AA3377', 'marker': 'v'},
}
PROPOSAL_COLOR = {k: v['color'] for k, v in PROPOSAL_STYLE.items()}
PROPOSAL_LABEL = {
    'uniform':           'Uniform',
    'gaussiantailed':    'Gaussian-tailed',
    'lineartailed':      'Linear-tailed',
    'exponentialtailed': 'Exp-tailed',
    'uniformtailed':     'Uniform-tailed',
}

RC_PARAMS = {
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'font.size': 9,
    'axes.titlesize': 9,
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_name', default='gaussianlinear',
                        choices=['gaussianlinear', 'gaussianlinearuniform'])
    parser.add_argument('--models_root', default=None)
    parser.add_argument('--out_root',    default=None)
    parser.add_argument('--n_samples',   type=int, default=2000)
    parser.add_argument('--grid_size',   type=int, default=10)
    args = parser.parse_args()
    if args.models_root is None:
        args.models_root = os.path.join(ROOT, 'results', args.task_name, 'exp1-dims', 'dim2')
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', args.task_name, '2d-analysis')
    os.makedirs(args.out_root, exist_ok=True)

    dim = 2
    if args.task_name == 'gaussianlinear':
        task = GaussianLinear(dim=dim, prior_scale=1.0)
    else:
        task = GaussianLinearUniform(dim=dim)
    simulator = task.get_simulator()

    posteriors = {}
    for name in PROPOSALS:
        pkl = os.path.join(args.models_root, name, 'posterior.pkl')
        if os.path.exists(pkl):
            posteriors[name] = load_posterior(pkl)
            print(f'Loaded {name}')
        else:
            print(f'WARNING: missing {pkl}')

    # ------------------------------------------------------------------ #
    # 1. Heatmap — 10x10 grid, C2ST(proposal, reference) at each point   #
    # ------------------------------------------------------------------ #
    cache_path = os.path.join(args.out_root, 'heatmap_cache.pkl')
    g = args.grid_size
    grid_vals  = np.linspace(-1.0, 1.0, g)
    grid_theta = np.array([[x, y] for y in grid_vals for x in grid_vals])

    if os.path.exists(cache_path):
        print('Loading cached heatmap samples')
        with open(cache_path, 'rb') as f:
            heatmap_cache = pickle.load(f)
    else:
        heatmap_cache = {'reference': [], 'posteriors': {n: [] for n in posteriors}}
        print('Sampling heatmap posteriors...')
        for theta in grid_theta:
            x_obs = simulator(torch.tensor(theta, dtype=torch.float32))
            if hasattr(task, '_get_reference_posterior'):
                ref = task._get_reference_posterior(observation=x_obs.unsqueeze(0))
                heatmap_cache['reference'].append(ref.sample((args.n_samples,)).cpu().numpy())
            else:
                s = task._sample_reference_posterior(args.n_samples, observation=x_obs.unsqueeze(0))
                heatmap_cache['reference'].append(s.reshape(args.n_samples, dim).cpu().numpy())
            for name, post in posteriors.items():
                s = post.sample((args.n_samples,), x_obs).cpu().numpy()
                heatmap_cache['posteriors'][name].append(s)
        with open(cache_path, 'wb') as f:
            pickle.dump(heatmap_cache, f)
        print(f'Saved heatmap cache to {cache_path}')

    # Compute C2ST-Hamming grids
    c2st_flat = {}; c2st_grids = {}
    for name in posteriors:
        scores = []
        for i in range(len(grid_theta)):
            ref = heatmap_cache['reference'][i]
            npe = heatmap_cache['posteriors'][name][i]
            scores.append(c2st(npe, ref))
        c2st_flat[name]  = np.array(scores)
        c2st_grids[name] = c2st_flat[name].reshape(g, g)

    # ── heatmap: 1 row (C2ST-Hamming) × n_proposals columns ─────────────────
    plt.rcParams.update(RC_PARAMS)
    loaded  = [n for n in PROPOSALS if n in c2st_grids]
    n_props = len(loaded)
    ticks   = np.linspace(-1.0, 1.0, g)
    c2st_vmax = max(c2st_grids[n].max() for n in loaded)

    fig, axes = plt.subplots(1, n_props,
                             figsize=(1.85 * n_props + 0.5, 2.2),
                             constrained_layout=True)
    if n_props == 1:
        axes = [axes]

    for col, name in enumerate(loaded):
        ax   = axes[col]
        grid = c2st_grids[name]
        im   = ax.imshow(grid, cmap='YlOrRd', vmin=0.0, vmax=min(float(c2st_vmax), 0.5),
                         extent=[-1, 1, -1, 1], origin='lower', aspect='equal',
                         interpolation='nearest')
        threshold = 0.3
        for r in range(g):
            for c_ in range(g):
                val = grid[r, c_]
                ax.text(ticks[c_], ticks[r], f'{val:.2f}',
                        ha='center', va='center', fontsize=3.5,
                        color='white' if val > threshold else 'black',
                        fontfamily='monospace')
        for cx, cy in CORNERS.values():
            ax.plot(cx, cy, 'k+', markersize=4, markeredgewidth=0.8)
        ax.tick_params(labelsize=6)
        ax.set_title(PROPOSAL_LABEL[name], fontsize=7.5, pad=3)
        ax.set_xlabel('θ₁', labelpad=2, fontsize=7)
        if col == 0:
            ax.set_ylabel('θ₂', labelpad=2, fontsize=7)
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=25)
    cbar.set_label('C2ST-Hamming\n0.5 = identical  ▸  lower = worse', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.axhline(0.5, color='k', linewidth=0.8)

    fig.suptitle(f'C2ST-Hamming vs reference — {args.task_name}', fontsize=9)
    hmap_path = os.path.join(args.out_root, 'heatmap_c2st.pdf')
    fig.savefig(hmap_path, bbox_inches='tight')
    fig.savefig(hmap_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {hmap_path}')
    plt.close(fig)

    # ── Radial plot ───────────────────────────────────────────────────────────
    radii      = np.sqrt((grid_theta ** 2).sum(axis=1))
    n_inner    = 5
    r_ie       = np.linspace(0.0, 1.0, n_inner + 1)
    bin_masks  = [(radii >= r_ie[b]) & (radii < r_ie[b + 1]) for b in range(n_inner)]
    bin_masks += [radii >= 1.0]
    bin_labels = [f'r={0.5*(r_ie[b]+r_ie[b+1]):.2f}' for b in range(n_inner)] + ['2σ extrap']
    n_bins     = len(bin_masks)
    x_pos      = np.arange(n_bins)
    offsets    = np.linspace(-0.2, 0.2, len(loaded))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5), constrained_layout=True)
    for i, name in enumerate(loaded):
        scores = c2st_flat[name]
        means  = np.array([scores[m].mean() if m.sum() else np.nan for m in bin_masks])
        errs   = np.array([scores[m].std()  if m.sum() > 1 else 0.0 for m in bin_masks])
        style  = PROPOSAL_STYLE[name]
        ax.errorbar(x_pos + offsets[i], means, yerr=errs,
                    fmt=style['marker'], color=style['color'],
                    capsize=2, markersize=4, linewidth=0, elinewidth=1.0,
                    label=PROPOSAL_LABEL[name], alpha=0.9, zorder=3)
    ax.axvline(n_inner - 0.5, color='#CC3311', linestyle='--', linewidth=1.0, alpha=0.8, zorder=0)
    ax.axhline(0.5, color='#888888', linestyle=':', linewidth=1.0)
    ax.set_ylabel('C2ST-Hamming vs Reference')
    ax.yaxis.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, n_bins - 0.5)
    ax.legend(fontsize=7.5, ncol=2, loc='lower left', handlelength=1.5)
    ax.set_title(f'Radial profiles — {args.task_name}', fontsize=9)
    ax.set_xticks(np.arange(n_bins + 1) - 0.5)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='major', length=3)
    ax.set_xticks(x_pos, minor=True)
    ax.set_xticklabels(bin_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlabel('Radius from origin')
    radial_path = os.path.join(args.out_root, 'radial_c2st.pdf')
    fig.savefig(radial_path, bbox_inches='tight')
    fig.savefig(radial_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {radial_path}')
    plt.close(fig)

    # ------------------------------------------------------------------ #
    # 2. Corner plots — 4 near-corner test points                         #
    # ------------------------------------------------------------------ #
    corner_cache_path = os.path.join(args.out_root, 'corner_cache.pkl')
    if os.path.exists(corner_cache_path):
        print('Loading cached corner samples')
        with open(corner_cache_path, 'rb') as f:
            corner_cache = pickle.load(f)
    else:
        corner_cache = {}
        for cname, (tx, ty) in CORNERS.items():
            theta = np.array([tx, ty])
            x_obs = simulator(torch.tensor(theta, dtype=torch.float32))
            if hasattr(task, '_get_reference_posterior'):
                ref = task._get_reference_posterior(observation=x_obs.unsqueeze(0))
                ref_samples = ref.sample((args.n_samples,)).cpu().numpy()
            else:
                s = task._sample_reference_posterior(args.n_samples, observation=x_obs.unsqueeze(0))
                ref_samples = s.reshape(args.n_samples, dim).cpu().numpy()
            corner_cache[cname] = {
                'theta': theta,
                'reference': ref_samples,
            }
            for name, post in posteriors.items():
                corner_cache[cname][name] = post.sample((args.n_samples,), x_obs).cpu().numpy()
        with open(corner_cache_path, 'wb') as f:
            pickle.dump(corner_cache, f)
        print(f'Saved corner cache to {corner_cache_path}')

    for cname, data in corner_cache.items():
        theta = data['theta']
        fig, axes = plt.subplots(2, 2, figsize=(4.5, 4.0))

        all_samples = {PROPOSAL_LABEL[n]: data[n] for n in posteriors if n in data}
        all_samples['Reference'] = data['reference']
        colors_map  = {PROPOSAL_LABEL[n]: PROPOSAL_COLOR[n] for n in posteriors}
        colors_map['Reference'] = '#888888'

        for i in range(2):
            ax = axes[i, i]
            for label, samps in all_samples.items():
                sns.kdeplot(data=samps[:, i], ax=ax, label=label,
                            color=colors_map[label], alpha=0.85, linewidth=1.4)
            ax.axvline(theta[i], color='#CC3311', linestyle='--', linewidth=1.2, label='True θ')
            ax.set_xlabel(f'θ{i+1}')
            if i == 0:
                ax.set_ylabel('Density')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        # Legend in the blank upper-right cell
        handles, labels_leg = axes[0, 0].get_legend_handles_labels()
        axes[0, 1].legend(handles, labels_leg, loc='center', ncol=1,
                          handlelength=1.2, frameon=False)

        ax = axes[1, 0]
        for label, samps in all_samples.items():
            sns.kdeplot(x=samps[:, 0], y=samps[:, 1], ax=ax,
                        color=colors_map[label], alpha=0.75,
                        levels=[0.05, 0.32], fill=False, linewidths=1.4)
        ax.scatter(theta[0], theta[1], color='#CC3311', s=60, marker='*', zorder=10)
        ax.axvline(theta[0], color='#CC3311', linestyle='--', linewidth=1.0, alpha=0.6)
        ax.axhline(theta[1], color='#CC3311', linestyle='--', linewidth=1.0, alpha=0.6)
        rect = mpatches.Rectangle((-1, -1), 2, 2, linewidth=1.2, edgecolor='#444444',
                                   facecolor='none', linestyle='-')
        ax.add_patch(rect)
        ax.set_xlim(-1.5, 1.5)
        ax.set_ylim(-1.5, 1.5)
        ax.set_xlabel('θ₁')
        ax.set_ylabel('θ₂')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        axes[0, 1].axis('off')

        fig.suptitle(f'θ = ({theta[0]}, {theta[1]})', fontsize=9)
        plt.tight_layout()
        corner_path = os.path.join(args.out_root, f'corner_{cname}.pdf')
        fig.savefig(corner_path, bbox_inches='tight')
        fig.savefig(corner_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f'Saved {corner_path}')
        plt.close(fig)

    print('\nDone.')


if __name__ == '__main__':
    main()
