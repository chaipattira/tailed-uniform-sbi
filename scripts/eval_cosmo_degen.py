"""
Spatial C2ST evaluation for the 2D cosmology (Omega_m, h) science experiment.

Loads trained NPE posteriors (from train_cosmo_2d.py), evaluates them on a
grid of (Om, h) test points against an MCMC reference, and produces:

  1. Heatmap  — |C2ST − 0.5| vs MCMC reference per proposal (one panel each)
  2. Corner plots — KDE overlays at near-corner test points in (Om, h) space

Reference samples come from either:
  • pre-computed MCMC grid in --mcmc_root (files: test_points.npy,
    mcmc_samples.npy, observations.npy)
  • inline emcee if --run_mcmc is passed (slow; ~1 min per grid point)

NPE posteriors are sampled in (phi1, phi2) rotated space and converted back to
(Om, h) via degen_rotation.inverse_batch before C2ST.

Usage
-----
python scripts/eval_cosmo_degen.py --run_mcmc          # build from scratch
python scripts/eval_cosmo_degen.py                     # use pre-cached grid
sbatch scripts/eval_cosmo_degen.sh
"""
import argparse
import os
import sys
import pickle

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.simulators import syren_simulator
from toolbox.transforms import degen_rotation
from toolbox.evaluators import c2st

OM_RANGE = (0.24, 0.40)
H_RANGE  = (0.61, 0.73)

PROPOSALS = ['uniform', 'gaussiantailed', 'lineartailed', 'exponentialtailed', 'uniformtailed']

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

# Near-corner test points inset to 80 % of prior width from center
_OM_C  = 0.5 * (OM_RANGE[0] + OM_RANGE[1])
_H_C   = 0.5 * (H_RANGE[0]  + H_RANGE[1])
_OM_D  = 0.4 * (OM_RANGE[1] - OM_RANGE[0])
_H_D   = 0.4 * (H_RANGE[1]  - H_RANGE[0])
CORNERS = {
    'ne': (_OM_C + _OM_D, _H_C + _H_D),
    'nw': (_OM_C - _OM_D, _H_C + _H_D),
    'se': (_OM_C + _OM_D, _H_C - _H_D),
    'sw': (_OM_C - _OM_D, _H_C - _H_D),
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

def _flatten_npe(x):
    """Support both old cache (single array) and new cache (list[n_repeats])."""
    if isinstance(x, list):
        return np.concatenate(x, axis=0)
    return x


def load_posterior(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def run_mcmc_single(x_obs, n_samples=1000, n_warmup=300, n_walkers=32):
    """emcee reference posterior for one (Om, h) observation."""
    import emcee
    import symbolic_pofk.syren_new as syren_new

    L, N = 1000, 128
    kf      = 2 * np.pi / L
    knyq    = np.pi * N / L
    kedges  = np.arange(0, knyq, kf)
    kc      = (kedges[:-1] + kedges[1:]) / 2

    def _log_prior(theta):
        Om, h = theta
        if not (OM_RANGE[0] <= Om <= OM_RANGE[1]):
            return -np.inf
        if not (H_RANGE[0] <= h <= H_RANGE[1]):
            return -np.inf
        return 0.0

    def _log_like(theta, xo):
        Om, h = theta
        Ob = 0.02242 / h**2
        P  = syren_new.pnl_new_emulated(kc, 2.105, Om, Ob, h, 0.9665, 0.0, -1.0, 0.0, a=1.0)
        Nk = L**3 * kc**2 * kf / (2 * np.pi**2)
        std = np.sqrt(np.abs(P)**2 * 2 / Nk)
        return -0.5 * np.sum(((xo - P) / std)**2) - np.sum(np.log(std))

    def _log_prob(theta, xo):
        lp = _log_prior(theta)
        return lp + _log_like(theta, xo) if np.isfinite(lp) else -np.inf

    pos = np.column_stack([
        np.random.uniform(*OM_RANGE, n_walkers),
        np.random.uniform(*H_RANGE,  n_walkers),
    ])
    sampler = emcee.EnsembleSampler(n_walkers, 2, _log_prob, args=(x_obs,))
    sampler.run_mcmc(pos, n_warmup, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, n_samples, progress=False)
    return sampler.get_chain(flat=True)   # (n_walkers * n_samples, 2)


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
    parser.add_argument('--models_root',   default=None)
    parser.add_argument('--mcmc_root',     default=None)
    parser.add_argument('--out_root',      default=None)
    parser.add_argument('--n_samples',     type=int, default=2000)
    parser.add_argument('--n_repeats',     type=int, default=10,
                        help='Number of NPE sampling repeats per obs; C2ST is averaged')
    parser.add_argument('--grid_size',     type=int, default=15)
    parser.add_argument('--mcmc_samples',  type=int, default=1000)
    parser.add_argument('--mcmc_warmup',   type=int, default=300)
    parser.add_argument('--mcmc_walkers',  type=int, default=32)
    parser.add_argument('--run_mcmc',      action='store_true',
                        help='Run emcee inline if no pre-computed reference found')
    parser.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=True,
                        help='Evaluate rotated (phi1,phi2) models (default) or original (Om,h) models')
    args = parser.parse_args()

    coord_subdir = 'rotated' if args.rotated else 'original'
    if args.models_root is None:
        args.models_root = os.path.join(ROOT, 'results', 'science', coord_subdir)
    if args.mcmc_root is None:
        args.mcmc_root = os.path.join(ROOT, 'mcmc_samples', 'grid')
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', 'science', coord_subdir, 'eval')
    os.makedirs(args.out_root, exist_ok=True)

    # ── load posteriors ───────────────────────────────────────────────────────
    posteriors = {}
    for name in PROPOSALS:
        pkl = os.path.join(args.models_root, name, 'posterior.pkl')
        if os.path.exists(pkl):
            posteriors[name] = load_posterior(pkl)
            print(f'Loaded {name}')
        else:
            print(f'WARNING: missing {pkl}')
    if not posteriors:
        raise RuntimeError('No posteriors found — run train_cosmo_2d.py first.')

    # ── heatmap grid ─────────────────────────────────────────────────────────
    cache_path = os.path.join(args.out_root, f'heatmap_cache_g{args.grid_size}_r{args.n_repeats}.pkl')
    if not os.path.exists(cache_path):
        _old = os.path.join(args.out_root, f'heatmap_cache_g{args.grid_size}.pkl')
        if os.path.exists(_old):
            cache_path = _old

    if os.path.exists(cache_path):
        print('Loading cached heatmap samples')
        with open(cache_path, 'rb') as f:
            hmap = pickle.load(f)
        grid_theta = hmap['grid_theta']
        g          = hmap['grid_size']
    else:
        # Build test grid
        g        = args.grid_size
        om_vals  = np.linspace(*OM_RANGE, g)
        h_vals   = np.linspace(*H_RANGE,  g)
        # row-major: outer loop over h (y-axis), inner over Om (x-axis)
        grid_theta = np.array([[om, h] for h in h_vals for om in om_vals])

        # Try loading pre-computed MCMC grid
        pts_npy  = os.path.join(args.mcmc_root, 'test_points.npy')
        obs_npy  = os.path.join(args.mcmc_root, 'observations.npy')
        ref_npy  = os.path.join(args.mcmc_root, 'mcmc_samples.npy')

        if os.path.exists(ref_npy) and os.path.exists(pts_npy):
            print(f'Using pre-computed MCMC from {args.mcmc_root}')
            grid_theta     = np.load(pts_npy)
            g              = int(round(np.sqrt(len(grid_theta))))
            observations   = np.load(obs_npy)
            reference_list = list(np.load(ref_npy))   # list of (n_ref, 2) arrays
        elif args.run_mcmc:
            print(f'Running inline MCMC on {len(grid_theta)} grid points...')
            observations   = []
            reference_list = []
            for theta in tqdm(grid_theta, desc='Grid MCMC'):
                x_obs = syren_simulator(theta)
                observations.append(x_obs)
                ref = run_mcmc_single(x_obs, args.mcmc_samples,
                                      args.mcmc_warmup, args.mcmc_walkers)
                reference_list.append(ref)
            observations = np.array(observations, dtype=np.float32)
        else:
            raise RuntimeError(
                'No MCMC reference found. Populate --mcmc_root or pass --run_mcmc.'
            )

        # Sample NPE posteriors at each grid point (n_repeats draws per obs)
        npe_samples = {name: [] for name in posteriors}
        print(f'Sampling NPE posteriors on grid (n_repeats={args.n_repeats})...')
        for x_obs in tqdm(observations, desc='NPE grid'):
            x_t = torch.tensor(x_obs, dtype=torch.float32)
            for name, post in posteriors.items():
                repeats = [
                    npe_sample_physical(post, x_t, args.n_samples, rotated=args.rotated)
                    for _ in range(args.n_repeats)
                ]
                npe_samples[name].append(repeats)  # list[n_repeats] of (n_samples, 2)

        hmap = {
            'grid_theta':  grid_theta,
            'grid_size':   g,
            'observations': observations,
            'reference':   reference_list,
            'posteriors':  npe_samples,
        }
        with open(cache_path, 'wb') as f:
            pickle.dump(hmap, f)
        print(f'Saved heatmap cache → {cache_path}')

    reference_list = hmap['reference']
    npe_samples    = hmap['posteriors']
    grid_theta     = hmap['grid_theta']

    # ── compute C2ST-Hamming grids ────────────────────────────────────────────
    c2st_flat  = {}; c2st_grids  = {}
    for name in posteriors:
        if name not in npe_samples:
            continue
        scores = []
        for i in range(len(grid_theta)):
            npe = _flatten_npe(npe_samples[name][i])
            ref = np.array(reference_list[i])
            scores.append(c2st(npe, ref))
        c2st_flat[name]  = np.array(scores)
        c2st_grids[name] = c2st_flat[name].reshape(g, g)

    # ── heatmap: 1 row (C2ST-Hamming) × n_proposals columns ─────────────────
    plt.rcParams.update(RC_PARAMS)
    loaded   = [n for n in PROPOSALS if n in c2st_grids]
    n_props  = len(loaded)
    om_ticks = np.linspace(OM_RANGE[0], OM_RANGE[1], g)
    h_ticks  = np.linspace(H_RANGE[0],  H_RANGE[1],  g)
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
                         extent=[OM_RANGE[0], OM_RANGE[1], H_RANGE[0], H_RANGE[1]],
                         origin='lower', aspect='auto', interpolation='nearest')
        threshold = 0.3
        for r in range(g):
            for c_ in range(g):
                val = grid[r, c_]
                ax.text(om_ticks[c_], h_ticks[r], f'{val:.2f}',
                        ha='center', va='center', fontsize=4.5,
                        color='white' if val > threshold else 'black',
                        fontfamily='monospace')
        for om, h in CORNERS.values():
            ax.plot(om, h, 'k+', markersize=4, markeredgewidth=0.8)
        ax.tick_params(labelsize=6)
        ax.set_title(PROPOSAL_LABEL[name], fontsize=7.5, pad=3)
        ax.set_xlabel('$\\Omega_m$', labelpad=2, fontsize=7)
        if col == 0:
            ax.set_ylabel('$h$', labelpad=2, fontsize=7)
        else:
            ax.set_yticklabels([])

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, aspect=25)
    cbar.set_label('C2ST-Hamming\n0.5 = identical  ▸  lower = worse', fontsize=7)
    cbar.ax.tick_params(labelsize=6)
    cbar.ax.axhline(0.5, color='k', linewidth=0.8)

    fig.suptitle('C2ST-Hamming vs MCMC reference  —  $(\\Omega_m,\\, h)$ grid', fontsize=9)
    hmap_path = os.path.join(args.out_root, 'heatmap_c2st_cosmo.pdf')
    fig.savefig(hmap_path, bbox_inches='tight')
    fig.savefig(hmap_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {hmap_path}')
    plt.close(fig)

    # ── phi2 and radial profiles: shared binning helper ───────────────────────
    rnames    = [n for n in PROPOSALS if n in c2st_flat]
    offsets_p = np.linspace(-0.2, 0.2, len(rnames))

    def _errorbars(ax, masks, offsets, x_pos, ylim=None, ideal_line=None):
        for i, name in enumerate(rnames):
            scores = c2st_flat[name]
            means  = np.array([scores[m].mean() if m.sum() else np.nan for m in masks])
            errs   = np.array([scores[m].std()  if m.sum() > 1 else 0.0 for m in masks])
            style  = PROPOSAL_STYLE[name]
            ax.errorbar(x_pos + offsets[i], means, yerr=errs,
                        fmt=style['marker'], color=style['color'],
                        capsize=2, markersize=4, linewidth=0, elinewidth=1.0,
                        label=PROPOSAL_LABEL[name], alpha=0.9, zorder=3)
        if ideal_line is not None:
            ax.axhline(ideal_line, color='#888888', linestyle=':', linewidth=1.0)
        if ylim is not None:
            ax.set_ylim(*ylim)
        ax.yaxis.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
        ax.set_axisbelow(True)

    # ── phi2 profile ──────────────────────────────────────────────────────────
    phi2_vals        = degen_rotation.forward_batch(grid_theta)[:, 1]
    phi2_lo, phi2_hi = phi2_vals.min(), phi2_vals.max()
    n_phi2_bins      = 5
    phi2_edges       = np.linspace(phi2_lo, phi2_hi + 1e-9, n_phi2_bins + 1)
    phi2_mids        = 0.5 * (phi2_edges[:-1] + phi2_edges[1:])
    phi2_masks       = [(phi2_vals >= phi2_edges[b]) & (phi2_vals < phi2_edges[b + 1])
                        for b in range(n_phi2_bins)]
    phi2_labels      = [f'$\\varphi_2$={m:.3f}' for m in phi2_mids]
    x_phi2           = np.arange(n_phi2_bins)

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5), constrained_layout=True)
    _errorbars(ax, phi2_masks, offsets_p, x_phi2, ideal_line=0.5)
    ax.axvspan(-0.5, 0.5, color='#DDDDDD', alpha=0.35, zorder=0)
    ax.axvspan(n_phi2_bins - 1.5, n_phi2_bins - 0.5, color='#DDDDDD', alpha=0.35, zorder=0)
    ax.set_ylabel('C2ST-Hamming vs MCMC')
    ax.set_xlim(-0.5, n_phi2_bins - 0.5)
    ax.legend(fontsize=7.5, ncol=2, loc='upper left', handlelength=1.5)
    ax.set_title('Profiles along degeneracy direction — $(\\Omega_m,\\, h)$ cosmology', fontsize=9)
    ax.set_xticks(np.arange(n_phi2_bins + 1) - 0.5)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='major', length=3)
    ax.set_xticks(x_phi2, minor=True)
    ax.set_xticklabels(phi2_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlabel('Degeneracy direction ($\\varphi_2$)')
    phi2_path = os.path.join(args.out_root, 'phi2_c2st_cosmo.pdf')
    fig.savefig(phi2_path, bbox_inches='tight')
    fig.savefig(phi2_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {phi2_path}')
    plt.close(fig)

    # ── Radial profile ────────────────────────────────────────────────────────
    if args.rotated:
        phi_grid = degen_rotation.forward_batch(grid_theta)
        (phi1_lo, phi1_hi), (phi2_lo, phi2_hi) = degen_rotation.prior_bounds_rotated()
        norm_x = (phi_grid[:, 0] - phi1_lo) / (phi1_hi - phi1_lo)
        norm_y = (phi_grid[:, 1] - phi2_lo) / (phi2_hi - phi2_lo)
        radial_xlabel = r'Radius from prior centre ($\phi_1$, $\phi_2$ normalised)'
        radial_title  = r'Radial profiles — cosmology, rotated $(\phi_1,\,\phi_2)$ coords'
    else:
        norm_x = (grid_theta[:, 0] - OM_RANGE[0]) / (OM_RANGE[1] - OM_RANGE[0])
        norm_y = (grid_theta[:, 1] - H_RANGE[0])  / (H_RANGE[1]  - H_RANGE[0])
        radial_xlabel = r'Radius from prior centre ($\Omega_m$, $h$ normalised)'
        radial_title  = r'Radial profiles — cosmology, physical $(\Omega_m,\,h)$ coords'

    radii    = np.sqrt((norm_x - 0.5)**2 + (norm_y - 0.5)**2)
    radii   /= radii.max()
    n_rbins  = 10
    r_edges  = np.linspace(0.0, 1.0 + 1e-9, n_rbins + 1)
    r_mids   = 0.5 * (r_edges[:-1] + r_edges[1:])
    r_masks  = [(radii >= r_edges[b]) & (radii < r_edges[b + 1]) for b in range(n_rbins)]
    r_labels = [f'r={m:.2f}' for m in r_mids]
    x_rad    = np.arange(n_rbins)
    offsets_r = np.linspace(-0.2, 0.2, len(rnames))

    fig, ax = plt.subplots(1, 1, figsize=(5.5, 2.5), constrained_layout=True)
    _errorbars(ax, r_masks, offsets_r, x_rad, ideal_line=0.5)
    ax.set_ylabel('C2ST-Hamming vs MCMC')
    ax.set_xlim(-0.5, n_rbins - 0.5)
    ax.legend(fontsize=7.5, ncol=2, loc='upper left', handlelength=1.5)
    ax.set_title(radial_title, fontsize=9)
    ax.set_xticks(np.arange(n_rbins + 1) - 0.5)
    ax.set_xticklabels([])
    ax.tick_params(axis='x', which='major', length=3)
    ax.set_xticks(x_rad, minor=True)
    ax.set_xticklabels(r_labels, minor=True, rotation=35, ha='right', fontsize=7.5)
    ax.tick_params(axis='x', which='minor', length=0)
    ax.set_xlabel(radial_xlabel)
    radial_path = os.path.join(args.out_root, 'radial_c2st_cosmo.pdf')
    fig.savefig(radial_path, bbox_inches='tight')
    fig.savefig(radial_path.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
    print(f'Saved {radial_path}')
    plt.close(fig)

    # ── corner plots ──────────────────────────────────────────────────────────
    corner_cache_path = os.path.join(args.out_root, 'corner_cache.pkl')

    if os.path.exists(corner_cache_path):
        print('Loading cached corner samples')
        with open(corner_cache_path, 'rb') as f:
            corner_cache = pickle.load(f)
    else:
        corner_cache = {}
        for cname, (om_c, h_c) in CORNERS.items():
            theta = np.array([om_c, h_c])
            x_obs = syren_simulator(theta).astype(np.float32)

            if args.run_mcmc:
                ref_samples = run_mcmc_single(x_obs, args.mcmc_samples,
                                              args.mcmc_warmup, args.mcmc_walkers)
            else:
                ref_samples = None

            x_t = torch.tensor(x_obs, dtype=torch.float32)
            entry = {'theta': theta, 'reference': ref_samples}
            for name, post in posteriors.items():
                entry[name] = npe_sample_physical(post, x_t, args.n_samples,
                                                   rotated=args.rotated)
            corner_cache[cname] = entry

        with open(corner_cache_path, 'wb') as f:
            pickle.dump(corner_cache, f)
        print(f'Saved corner cache → {corner_cache_path}')

    for cname, data in corner_cache.items():
        theta       = data['theta']
        ref_samples = data.get('reference')

        all_samples = {PROPOSAL_LABEL[n]: data[n] for n in posteriors if n in data}
        colors_map  = {PROPOSAL_LABEL[n]: PROPOSAL_COLOR[n] for n in posteriors}
        if ref_samples is not None:
            all_samples['MCMC ref'] = ref_samples
            colors_map['MCMC ref']  = '#888888'

        fig, axes = plt.subplots(2, 2, figsize=(4.5, 4.0))
        param_names = ['$\\Omega_m$', '$h$']

        for i in range(2):
            ax = axes[i, i]
            for label, samps in all_samples.items():
                lw = 1.8 if label == 'MCMC ref' else 1.4
                ls = '--' if label == 'MCMC ref' else '-'
                sns.kdeplot(data=samps[:, i], ax=ax, label=label,
                            color=colors_map[label], alpha=0.85,
                            linewidth=lw, linestyle=ls)
            ax.axvline(theta[i], color='#CC3311', linestyle='--',
                       linewidth=1.2, label='True θ')
            ax.set_xlabel(param_names[i])
            if i == 0:
                ax.set_ylabel('Density')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

        handles, labels_leg = axes[0, 0].get_legend_handles_labels()
        axes[0, 1].legend(handles, labels_leg, loc='center', ncol=1,
                          handlelength=1.2, frameon=False, fontsize=7.5)
        axes[0, 1].axis('off')

        ax2d = axes[1, 0]
        for label, samps in all_samples.items():
            lw = 1.8 if label == 'MCMC ref' else 1.4
            ls = '--' if label == 'MCMC ref' else '-'
            sns.kdeplot(x=samps[:, 0], y=samps[:, 1], ax=ax2d,
                        color=colors_map[label], alpha=0.75,
                        levels=[0.05, 0.32], fill=False, linewidths=lw,
                        linestyles=ls)
        ax2d.scatter(theta[0], theta[1], color='#CC3311', s=60, marker='*', zorder=10)
        ax2d.axvline(theta[0], color='#CC3311', linestyle='--', linewidth=1.0, alpha=0.6)
        ax2d.axhline(theta[1], color='#CC3311', linestyle='--', linewidth=1.0, alpha=0.6)
        rect = mpatches.Rectangle(
            (OM_RANGE[0], H_RANGE[0]),
            OM_RANGE[1] - OM_RANGE[0], H_RANGE[1] - H_RANGE[0],
            linewidth=1.2, edgecolor='#444444', facecolor='none',
        )
        ax2d.add_patch(rect)
        ax2d.set_xlim(OM_RANGE[0] - 0.03, OM_RANGE[1] + 0.03)
        ax2d.set_ylim(H_RANGE[0]  - 0.03, H_RANGE[1]  + 0.03)
        ax2d.set_xlabel('$\\Omega_m$')
        ax2d.set_ylabel('$h$')
        ax2d.spines['top'].set_visible(False)
        ax2d.spines['right'].set_visible(False)

        fig.suptitle(f'$\\Omega_m={theta[0]:.3f},\\ h={theta[1]:.3f}$', fontsize=9)
        plt.tight_layout()
        cpath = os.path.join(args.out_root, f'corner_{cname}.pdf')
        fig.savefig(cpath, bbox_inches='tight')
        fig.savefig(cpath.replace('.pdf', '.png'), dpi=300, bbox_inches='tight')
        print(f'Saved {cpath}')
        plt.close(fig)

    print('\nDone.')


if __name__ == '__main__':
    main()
