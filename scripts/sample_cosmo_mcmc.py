"""
Generates MCMC reference samples per grid point. NPE sampling is done
downstream in eval_cosmo.

For each grid point theta=(Om,h), draws n_obs noisy observations from
syren_simulator and runs MCMC conditioned on each.

Saved per-task pkl contains:
    theta     : (2,)                         — [Om, h]
    task_id   : int
    grid_size : int
    n_obs     : int
    rotated   : bool
    obs_list  : list[n_obs] of (n_k,)        — noisy P(k) realizations
    mcmc_list : list[n_obs] of (n_mcmc*n_walkers, 2)

Output: results/science/{rotated|original}/grid_samples/point_{task_id:04d}.pkl

Usage
-----
python scripts/sample_cosmo_mcmc.py --task_id 0
sbatch scripts/sample_cosmo_mcmc.sh
"""
import argparse
import os
import sys
import pickle

import numpy as np
import torch
import emcee

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.simulators import syren_simulator

OM_RANGE  = (0.24, 0.40)   # physical bounds — used for evaluation grid
H_RANGE   = (0.61, 0.73)

# Log-Gaussian MCMC prior: log(theta) ~ Normal(mu_log, sigma_log^2), truncated at ±2σ
# ±3σ support — explicitly chosen to be tighter than the full physical range
_MCMC_OM_RANGE = (0.27, 0.37)
_MCMC_H_RANGE  = (0.63, 0.71)

_MU_LOG_OM    = np.log((_MCMC_OM_RANGE[0] + _MCMC_OM_RANGE[1]) / 2)
_SIGMA_LOG_OM = np.log(_MCMC_OM_RANGE[1] / _MCMC_OM_RANGE[0]) / 6   # ±3σ = range bounds
_MU_LOG_H     = np.log((_MCMC_H_RANGE[0]  + _MCMC_H_RANGE[1])  / 2)
_SIGMA_LOG_H  = np.log(_MCMC_H_RANGE[1]  / _MCMC_H_RANGE[0])  / 6


def run_mcmc_single(x_obs, n_samples=1000, n_warmup=300, n_walkers=32):
    import symbolic_pofk.syren_new as syren_new
    L, N = 1000, 128
    kf   = 2 * np.pi / L
    knyq = np.pi * N / L
    kc   = (np.arange(0, knyq, kf)[:-1] + np.arange(0, knyq, kf)[1:]) / 2

    def _log_prior(theta):
        Om, h = theta
        if not (_MCMC_OM_RANGE[0] <= Om <= _MCMC_OM_RANGE[1]): return -np.inf
        if not (_MCMC_H_RANGE[0]  <= h  <= _MCMC_H_RANGE[1]):  return -np.inf
        lp  = -0.5 * ((np.log(Om) - _MU_LOG_OM) / _SIGMA_LOG_OM)**2 - np.log(Om)
        lp += -0.5 * ((np.log(h)  - _MU_LOG_H)  / _SIGMA_LOG_H )**2 - np.log(h)
        return lp

    def _log_like(theta, xo):
        Om, h = theta
        P   = syren_new.pnl_new_emulated(kc, 2.105, Om, 0.02242/h**2, h, 0.9665, 0.0, -1.0, 0.0, a=1.0)
        Nk  = L**3 * kc**2 * kf / (2 * np.pi**2)
        std = np.sqrt(np.abs(P)**2 * 2 / Nk)
        return -0.5 * np.sum(((xo - P) / std)**2) - np.sum(np.log(std))

    def _log_prob(theta, xo):
        lp = _log_prior(theta)
        return lp + _log_like(theta, xo) if np.isfinite(lp) else -np.inf

    pos = np.column_stack([
        np.random.uniform(*_MCMC_OM_RANGE, n_walkers),
        np.random.uniform(*_MCMC_H_RANGE,  n_walkers),
    ])
    sampler = emcee.EnsembleSampler(n_walkers, 2, _log_prob, args=(x_obs,))
    sampler.run_mcmc(pos, n_warmup, progress=False)
    sampler.reset()
    sampler.run_mcmc(None, n_samples, progress=False)
    return sampler.get_chain(flat=True)   # (n_walkers * n_samples, 2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id',         type=int, required=True)
    parser.add_argument('--grid_size',       type=int, default=10)
    parser.add_argument('--n_obs',           type=int, default=5)
    parser.add_argument('--n_mcmc_samples',  type=int, default=1000)
    parser.add_argument('--n_mcmc_warmup',   type=int, default=300)
    parser.add_argument('--n_mcmc_walkers',  type=int, default=32)
    parser.add_argument('--out_root',        default=None)
    parser.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    coord_subdir = 'rotated' if args.rotated else 'original'
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', 'science', coord_subdir, 'grid_samples')
    os.makedirs(args.out_root, exist_ok=True)

    g = args.grid_size
    if not (0 <= args.task_id < g * g):
        raise ValueError(f'task_id must be 0-{g*g-1}')

    out_path = os.path.join(args.out_root, f'point_{args.task_id:04d}.pkl')
    if os.path.exists(out_path):
        print(f'Already exists, skipping: {out_path}')
        return

    # Row-major grid: outer loop h, inner loop Om (matches eval_cosmo_degen.py)
    om_vals    = np.linspace(*_MCMC_OM_RANGE, g)
    h_vals     = np.linspace(*_MCMC_H_RANGE,  g)
    grid_theta = np.array([[om, h] for h in h_vals for om in om_vals])
    theta      = grid_theta[args.task_id]

    np.random.seed(args.task_id)
    torch.manual_seed(args.task_id)

    print(f'[task {args.task_id}/{g*g-1}] Om={theta[0]:.4f}  h={theta[1]:.4f}  n_obs={args.n_obs}')

    obs_list  = []
    mcmc_list = []

    for i in range(args.n_obs):
        print(f'  realization {i+1}/{args.n_obs}')

        x_obs = syren_simulator(theta).astype(np.float32)
        obs_list.append(x_obs)

        ref = run_mcmc_single(x_obs, args.n_mcmc_samples,
                              args.n_mcmc_warmup, args.n_mcmc_walkers)
        mcmc_list.append(ref)
        print(f'    MCMC done: {ref.shape}  Om={ref[:,0].mean():.4f}±{ref[:,0].std():.4f}  h={ref[:,1].mean():.4f}±{ref[:,1].std():.4f}')

    result = {
        'theta':    theta,
        'task_id':  args.task_id,
        'grid_size': g,
        'n_obs':    args.n_obs,
        'rotated':  args.rotated,
        'obs_list':  obs_list,
        'mcmc_list': mcmc_list,
    }
    with open(out_path, 'wb') as f:
        pickle.dump(result, f)
    print(f'  saved → {out_path}')


if __name__ == '__main__':
    main()
