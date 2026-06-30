"""
Spatial C2ST evaluation for the 2D cosmology (Omega_m, h) science experiment.

Reads MCMC reference samples from sample_cosmo_mcmc output (a directory of
point_XXXX.pkl files, each containing obs_list and mcmc_list).  Samples NPE
posteriors inline (sequential, single node), computes C2ST averaged over
cosmic-variance realizations at each grid point, and produces:

  1. Heatmap  — |C2ST − 0.5| vs MCMC reference per proposal
  2. phi2 profile  — C2ST along the degeneracy direction
  3. phi2-wall distance profile  — C2ST vs distance to phi2 prior boundary
  4. Radial profile  — C2ST vs radial distance from prior centre

Results are cached to a pkl so plots can be regenerated without rerunning
NPE sampling.

Usage
-----
python scripts/eval_cosmo.py
python scripts/eval_cosmo.py --no-rotated
python scripts/eval_cosmo.py --grid_size 10 --samples_root path/to/pkls
sbatch scripts/eval_cosmo.sh
"""
import argparse
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.transforms import degen_rotation
from toolbox.evaluators import c2st

# MCMC prior support — matches sample_cosmo_mcmc.py
OM_RANGE = (0.27, 0.37)
H_RANGE  = (0.63, 0.71)

PROPOSALS = ['uniform', 'gaussiantailed', 'lineartailed', 'exponentialtailed', 'uniformtailed']

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


# ── helpers ───────────────────────────────────────────────────────────────────

def load_posterior(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def npe_sample_physical(post, x_obs_tensor, n_samples, rotated=True):
    """Sample from an NPE posterior and return (Om, h) samples.

    rotated=True  — posterior lives in (phi1, phi2); apply inverse rotation.
    rotated=False — posterior lives in (Om, h) directly; no rotation needed.
    """
    samples = post.sample((n_samples,), x_obs_tensor).cpu().numpy()  # (n, 2)
    if rotated:
        return degen_rotation.inverse_batch(samples)  # [phi1,phi2] -> [Om,h]
    return samples                                    # already [Om, h]


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--grid_size',     type=int, default=15,
                        help='Grid size used by sample_cosmo_mcmc (default 15)')
    parser.add_argument('--samples_root',  default=None,
                        help='Directory of point_XXXX.pkl from sample_cosmo_mcmc')
    parser.add_argument('--models_root',   default=None,
                        help='Directory of trained posteriors (one subdir per proposal)')
    parser.add_argument('--out_root',      default=None,
                        help='Where to save plots and C2ST cache pkl')
    parser.add_argument('--n_npe_samples', type=int, default=2000,
                        help='Number of samples to draw from each NPE per observation')
    parser.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=True,
                        help='Models trained in (phi1,phi2) rotated coords (default) '
                             'or original (Om,h) coords')
    args = parser.parse_args()

    coord_subdir = 'rotated' if args.rotated else 'original'
    if args.samples_root is None:
        args.samples_root = os.path.join(
            ROOT, 'results', 'science', coord_subdir, 'grid_samples'
        )
    if args.models_root is None:
        args.models_root = os.path.join(ROOT, 'results', 'science', coord_subdir)
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', 'science', coord_subdir, 'eval')
    os.makedirs(args.out_root, exist_ok=True)

    g         = args.grid_size
    n_points  = g * g
    om_vals   = np.linspace(*OM_RANGE, g)
    h_vals    = np.linspace(*H_RANGE,  g)
    grid_theta = np.array([[om, h] for h in h_vals for om in om_vals])

    # ── load MCMC cache pkls ──────────────────────────────────────────────────
    print(f'Loading {n_points} grid-point pkls from {args.samples_root} ...')
    records = {}
    missing = []
    for tid in range(n_points):
        path = os.path.join(args.samples_root, f'point_{tid:04d}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                records[tid] = pickle.load(f)
        else:
            missing.append(tid)

    if missing:
        raise RuntimeError(
            f'{len(missing)} grid point(s) missing: task_ids {missing[:10]}'
            f'{"..." if len(missing) > 10 else ""}\n'
            f'Run sample_cosmo_mcmc.sh first.'
        )

    n_obs = records[0]['n_obs']
    print(f'Loaded all {n_points} points  (n_obs={n_obs} per point)')

    # ── load trained NPE posteriors ───────────────────────────────────────────
    posteriors = {}
    for name in PROPOSALS:
        pkl = os.path.join(args.models_root, name, 'posterior.pkl')
        if os.path.exists(pkl):
            posteriors[name] = load_posterior(pkl)
            print(f'Loaded posterior: {name}')
        else:
            print(f'WARNING: missing {pkl}')
    if not posteriors:
        raise RuntimeError('No posteriors found — run train_cosmo_2d.py first.')

    present_proposals = [p for p in PROPOSALS if p in posteriors]

    # ── compute C2ST (or load from cache) ────────────────────────────────────
    cache_path = os.path.join(args.out_root, f'c2st_cache_g{g}_nobs{n_obs}.pkl')

    if os.path.exists(cache_path):
        print(f'Loading cached C2ST results from {cache_path}')
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        c2st_flat = cache['c2st_flat']
        n_obs     = cache['n_obs']
    else:
        # Sample NPE posteriors and compute C2ST at each grid point
        c2st_flat = {name: np.zeros(n_points) for name in present_proposals}

        print(f'Sampling NPE and computing C2ST  '
              f'(n_npe_samples={args.n_npe_samples}, n_obs={n_obs}) ...')
        for tid in tqdm(range(n_points), desc='Grid points'):
            rec = records[tid]
            for name in present_proposals:
                post = posteriors[name]
                scores = []
                for j in range(n_obs):
                    x_t = torch.tensor(rec['obs_list'][j], dtype=torch.float32)
                    npe_samps = npe_sample_physical(
                        post, x_t, args.n_npe_samples, rotated=args.rotated
                    )
                    scores.append(
                        c2st(npe_samps, rec['mcmc_list'][j], seed=42 + j)
                    )
                c2st_flat[name][tid] = np.mean(scores)

        cache = {
            'c2st_flat':  c2st_flat,
            'grid_theta': grid_theta,
            'grid_size':  g,
            'n_obs':      n_obs,
            'rotated':    args.rotated,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(cache, f)
        print(f'Saved C2ST cache -> {cache_path}')

    loaded     = [n for n in PROPOSALS if n in c2st_flat]
    c2st_grids = {name: c2st_flat[name].reshape(g, g) for name in loaded}
    abs_grids  = {name: np.abs(c2st_grids[name] - 0.5) for name in loaded}

    plt.rcParams.update(RC_PARAMS)
    om_ticks = np.linspace(*OM_RANGE, g)
    h_ticks  = np.linspace(*H_RANGE,  g)

    # ── heatmap: |C2ST − 0.5| ────────────────────────────────────────────────
    fig, axes = plt.subplots(len(loaded), 1,
                             figsize=(4.0, 3.0 * len(loaded) + 0.5),
                             constrained_layout=True)
    if len(loaded) == 1:
        axes = [axes]

    im = None
    for ax, name in zip(axes, loaded):
        im = ax.imshow(abs_grids[name], cmap='YlOrRd', vmin=0.0, vmax=0.15,
                       extent=[OM_RANGE[0], OM_RANGE[1], H_RANGE[0], H_RANGE[1]],
                       origin='lower', aspect='auto', interpolation='nearest')
        for row in range(g):
            for col in range(g):
                val = abs_grids[name][row, col]
                ax.text(om_ticks[col], h_ticks[row], f'{val:.2f}',
                        ha='center', va='center', fontsize=5.0,
                        color='white' if val > 0.10 else 'black',
                        fontfamily='monospace')
        ax.set_title(PROPOSAL_LABEL[name], fontsize=8, pad=3)
        ax.set_ylabel('$h$', labelpad=2)
        ax.tick_params(labelsize=7)

    axes[-1].set_xlabel('$\\Omega_m$', labelpad=2)
    for ax in axes[:-1]:
        ax.set_xticklabels([])

    if im is not None:
        cbar = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03, aspect=40)
        cbar.set_label(
            f'|C2ST − 0.5| vs MCMC (avg {n_obs} realizations)\n'
            '0 = perfect  ▸  higher = worse',
            fontsize=7.5,
        )
        cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        f'|C2ST − 0.5| vs MCMC  —  $(\\Omega_m,\\,h)$ grid  (n_obs={n_obs})',
        fontsize=9,
    )
    hmap_path = os.path.join(args.out_root, 'heatmap_absc2st_cosmo.pdf')
    fig.savefig(hmap_path, bbox_inches='tight')
    fig.savefig(hmap_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {hmap_path}')
    plt.close(fig)

    # ── phi2 profile ──────────────────────────────────────────────────────────
    phi2_vals   = degen_rotation.forward_batch(grid_theta)[:, 1]
    n_phi2_bins = 10
    phi2_edges  = np.linspace(phi2_vals.min(), phi2_vals.max() + 1e-9, n_phi2_bins + 1)
    phi2_mids   = 0.5 * (phi2_edges[:-1] + phi2_edges[1:])
    phi2_masks  = [
        (phi2_vals >= phi2_edges[b]) & (phi2_vals < phi2_edges[b + 1])
        for b in range(n_phi2_bins)
    ]
    phi2_labels = [f'$\\varphi_2$={m:.3f}' for m in phi2_mids]
    offsets     = np.linspace(-0.2, 0.2, len(loaded))

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    for i, name in enumerate(loaded):
        scores = c2st_flat[name]
        means  = [scores[m].mean() if m.sum() else np.nan for m in phi2_masks]
        errs   = [scores[m].std()  if m.sum() > 1 else 0.0 for m in phi2_masks]
        style  = PROPOSAL_STYLE[name]
        ax.errorbar(np.arange(n_phi2_bins) + offsets[i], means, yerr=errs,
                    fmt=style['marker'], color=style['color'],
                    capsize=2, markersize=4, linewidth=0, elinewidth=1.0,
                    label=PROPOSAL_LABEL[name], alpha=0.9, zorder=3)

    ax.axvspan(-0.5, 0.5, color='#DDDDDD', alpha=0.35, zorder=0,
               label='Near prior boundary')
    ax.axvspan(n_phi2_bins - 1.5, n_phi2_bins - 0.5, color='#DDDDDD', alpha=0.35, zorder=0)
    ax.axhline(0.5, color='#888888', linestyle=':', linewidth=1.0,
               label='C2ST = 0.5 (ideal)')
    ax.set_xticks(np.arange(n_phi2_bins + 1) - 0.5)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='major', length=3)
    ax.set_xticks(np.arange(n_phi2_bins), minor=True)
    ax.set_xticklabels(phi2_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlim(-0.5, n_phi2_bins - 0.5)
    ax.set_xlabel(
        'Degeneracy direction'
        r' ($\varphi_2 = -n_2\,\ln\Omega_m + n_1\,\ln h$)'
    )
    ax.set_ylabel('C2ST vs MCMC reference')
    ax.set_title(
        f'C2ST along degeneracy direction  (avg {n_obs} realizations)', fontsize=9
    )
    ax.legend(fontsize=7.5, ncol=2, loc='upper left', handlelength=1.5)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()
    phi2_path = os.path.join(args.out_root, 'phi2_c2st_cosmo.pdf')
    fig.savefig(phi2_path, bbox_inches='tight')
    fig.savefig(phi2_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {phi2_path}')
    plt.close(fig)

    # ── distance-to-phi2-wall profile ─────────────────────────────────────────
    _, (phi2_prior_lo, phi2_prior_hi) = degen_rotation.prior_bounds_rotated()
    phi2_range  = phi2_prior_hi - phi2_prior_lo
    dist_wall   = np.minimum(
        phi2_vals - phi2_prior_lo,
        phi2_prior_hi - phi2_vals,
    ) / phi2_range   # [0, 0.5]

    n_wall_bins = 6
    wall_edges  = np.linspace(0.0, 0.5 + 1e-9, n_wall_bins + 1)
    wall_mids   = 0.5 * (wall_edges[:-1] + wall_edges[1:])
    wall_masks  = [
        (dist_wall >= wall_edges[b]) & (dist_wall < wall_edges[b + 1])
        for b in range(n_wall_bins)
    ]
    wall_labels = [f'{m:.2f}' for m in wall_mids]
    offsets_w   = np.linspace(-0.2, 0.2, len(loaded))

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    x_pos = np.arange(n_wall_bins)
    for i, name in enumerate(loaded):
        scores = c2st_flat[name]
        means  = np.array([scores[m].mean() if m.sum() else np.nan for m in wall_masks])
        errs   = np.array([scores[m].std()  if m.sum() > 1 else 0.0  for m in wall_masks])
        style  = PROPOSAL_STYLE[name]
        ax.errorbar(x_pos + offsets_w[i], means, yerr=errs,
                    fmt=style['marker'], color=style['color'],
                    capsize=2, markersize=4, linewidth=0, elinewidth=1.0,
                    label=PROPOSAL_LABEL[name], alpha=0.9, zorder=3)

    ax.axvspan(-0.5, 0.5, color='#DDDDDD', alpha=0.40, zorder=0)
    ax.axhline(0.5, color='#888888', linestyle=':', linewidth=1.0,
               label='C2ST = 0.5 (ideal)')
    ax.set_xticks(np.arange(n_wall_bins + 1) - 0.5)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='major', length=3)
    ax.set_xticks(x_pos, minor=True)
    ax.set_xticklabels(wall_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlim(-0.5, n_wall_bins - 0.5)
    ax.set_xlabel(
        r'Normalised distance from $\varphi_2$ prior wall'
        r'  ($0$ = boundary, $0.5$ = centre)'
    )
    ax.set_ylabel('C2ST vs MCMC reference')
    ax.set_title(
        f'C2ST vs distance to $\\varphi_2$ boundary  '
        f'(avg {n_obs} realizations)',
        fontsize=9,
    )
    ax.legend(fontsize=7.5, ncol=2, loc='upper left', handlelength=1.5)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()
    wall_path = os.path.join(args.out_root, 'wall_dist_c2st_cosmo.pdf')
    fig.savefig(wall_path, bbox_inches='tight')
    fig.savefig(wall_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {wall_path}')
    plt.close(fig)

    # ── radial C2ST ───────────────────────────────────────────────────────────
    if args.rotated:
        phi_grid = degen_rotation.forward_batch(grid_theta)
        (phi1_lo, phi1_hi), (phi2_lo, phi2_hi) = degen_rotation.prior_bounds_rotated()
        norm_x = (phi_grid[:, 0] - phi1_lo) / (phi1_hi - phi1_lo)
        norm_y = (phi_grid[:, 1] - phi2_lo) / (phi2_hi - phi2_lo)
        radial_xlabel = r'Radius from prior centre ($\phi_1$, $\phi_2$ normalised)'
        radial_title  = r'Radial C2ST — cosmology, rotated $(\phi_1,\,\phi_2)$ coords'
    else:
        norm_x = (grid_theta[:, 0] - OM_RANGE[0]) / (OM_RANGE[1] - OM_RANGE[0])
        norm_y = (grid_theta[:, 1] - H_RANGE[0])  / (H_RANGE[1]  - H_RANGE[0])
        radial_xlabel = r'Radius from prior centre ($\Omega_m$, $h$ normalised)'
        radial_title  = r'Radial C2ST — cosmology, physical $(\Omega_m,\,h)$ coords'

    radii    = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
    radii   /= radii.max()
    n_rbins  = 8
    r_edges  = np.linspace(0.0, 1.0 + 1e-9, n_rbins + 1)
    r_mids   = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_masks  = [
        (radii >= r_edges[b]) & (radii < r_edges[b + 1])
        for b in range(n_rbins)
    ]
    r_labels  = [f'r={m:.2f}' for m in r_mids]
    offsets_r = np.linspace(-0.2, 0.2, len(loaded))

    fig, ax = plt.subplots(figsize=(5.5, 2.5))
    for i, name in enumerate(loaded):
        scores = c2st_flat[name]
        means  = [scores[m].mean() if m.sum() else np.nan for m in r_masks]
        errs   = [scores[m].std()  if m.sum() > 1 else 0.0 for m in r_masks]
        style  = PROPOSAL_STYLE[name]
        ax.errorbar(np.arange(n_rbins) + offsets_r[i], means, yerr=errs,
                    fmt=style['marker'], color=style['color'],
                    capsize=2, markersize=4, linewidth=0, elinewidth=1.0,
                    label=PROPOSAL_LABEL[name], alpha=0.9, zorder=3)

    ax.axhline(0.5, color='#888888', linestyle=':', linewidth=1.0,
               label='C2ST = 0.5 (ideal)')
    ax.set_xticks(np.arange(n_rbins + 1) - 0.5)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='major', length=3)
    ax.set_xticks(np.arange(n_rbins), minor=True)
    ax.set_xticklabels(r_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlim(-0.5, n_rbins - 0.5)
    ax.set_xlabel(radial_xlabel)
    ax.set_ylabel('C2ST vs MCMC reference')
    ax.set_title(
        f'{radial_title}  (avg {n_obs} realizations)', fontsize=9
    )
    ax.legend(fontsize=7.5, ncol=2, loc='upper left', handlelength=1.5)
    ax.yaxis.grid(True, alpha=0.15)
    ax.set_axisbelow(True)
    plt.tight_layout()
    radial_path = os.path.join(args.out_root, 'radial_c2st_cosmo.pdf')
    fig.savefig(radial_path, bbox_inches='tight')
    fig.savefig(radial_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {radial_path}')
    plt.close(fig)

    print('\nDone.')


if __name__ == '__main__':
    main()
