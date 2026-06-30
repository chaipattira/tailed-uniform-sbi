"""
Train one NPE model for the 2D cosmology (Omega_m, h) science experiment.

Two coordinate modes controlled by --rotated / --no-rotated:

  --rotated  (default)  Parameters in (phi1, phi2) aligned with the degeneracy
                        direction Omega_m * h^2.563 = const (degen_detector).
                        Output: results/science/rotated/{proposal}/

  --no-rotated          Parameters in original (Omega_m, h) space.
                        Output: results/science/original/{proposal}/

Proposals (task_id):
  0: uniform           — flat uniform
  1: gaussiantailed    — Gaussian-tailed
  2: lineartailed      — linear-tailed
  3: exponentialtailed — exponential-tailed
  4: uniformtailed     — uniform-tailed

Usage
-----
sbatch scripts/train_cosmo_2d.sh                    # rotated, array 0-4
sbatch scripts/train_cosmo_2d.sh --no-rotated       # original, array 0-4
python scripts/train_cosmo_2d.py --task_id 0 --rotated
python scripts/train_cosmo_2d.py --task_id 0 --no-rotated
"""
import argparse
import os
import sys
import random

import numpy as np
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from toolbox.distributions import (
    GaussianTailed, LinearTailed, ExponentialTailed, UniformTailed,
)
from toolbox.simulators import syren_simulator, sample_uniform_lhs
from toolbox.priors import get_rotated_priors
from toolbox.transforms import degen_rotation

PROPOSALS = [
    'uniform',
    'gaussiantailed',
    'lineartailed',
    'exponentialtailed',
    'uniformtailed',
]

SIGMA_SCALE = 0.1   # tail width as fraction of prior range

OM_RANGE = (0.27, 0.37) # ±3σ support
H_RANGE  = (0.63, 0.71) # ±3σ support


def run_simulations(theta_phys_np, desc='Simulating'):
    """Run syren_simulator over a batch of (Om, h) parameters.
    theta_phys_np : ndarray (n, 2) — columns are [Om, h]
    """
    xs = []
    for row in tqdm(theta_phys_np, desc=desc, leave=False):
        xs.append(syren_simulator(row))
    return np.array(xs, dtype=np.float32)   # (n, n_k)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help=f'0-{len(PROPOSALS)-1}: {PROPOSALS}')
    parser.add_argument('--n_sims', type=int, default=3000)
    parser.add_argument('--out_root', type=str, default=None)
    parser.add_argument('--sigma_scale', type=float, default=SIGMA_SCALE,
                        help='Tail width as fraction of prior range (default: %(default)s)')
    parser.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=True,
                        help='Train in rotated (phi1,phi2) space (default) or original (Om,h) space')
    args = parser.parse_args()

    coord_subdir = 'rotated' if args.rotated else 'original'
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', 'science', coord_subdir)

    if not (0 <= args.task_id < len(PROPOSALS)):
        raise ValueError(f'task_id must be 0-{len(PROPOSALS)-1}, got {args.task_id}')

    proposal_name = PROPOSALS[args.task_id]
    out_dir = os.path.join(args.out_root, proposal_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] proposal={proposal_name}  rotated={args.rotated}'
          f'  sigma_scale={args.sigma_scale}  n_sims={args.n_sims}  out={out_dir}')

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    cls_map = {
        'gaussiantailed':    GaussianTailed,
        'lineartailed':      LinearTailed,
        'exponentialtailed': ExponentialTailed,
        'uniformtailed':     UniformTailed,
    }

    if args.rotated:
        # ── rotated (phi1, phi2) space ────────────────────────────────────────
        _, _, phi1_range, phi2_range = get_rotated_priors(device=device, sigma_scale=args.sigma_scale)
        a   = [phi1_range[0], phi2_range[0]]
        b   = [phi1_range[1], phi2_range[1]]
        sigma = [args.sigma_scale * (phi1_range[1] - phi1_range[0]),
                 args.sigma_scale * (phi2_range[1] - phi2_range[0])]
        print(f'phi1 range: {phi1_range}')
        print(f'phi2 range: {phi2_range}')

        loc   = [(a[i] + b[i]) / 2 for i in range(len(a))]
        scale = [(b[i] - a[i]) / 2 for i in range(len(a))]
        prior_assumed = ili.utils.IndependentNormal(loc=loc, scale=scale, device=device)

        if proposal_name == 'uniform':
            proposal = prior_assumed
            theta = sample_uniform_lhs(args.n_sims, [phi1_range, phi2_range])
        else:
            proposal = cls_map[proposal_name](
                a=torch.tensor(a, dtype=torch.float32),
                b=torch.tensor(b, dtype=torch.float32),
                sigma=torch.tensor(sigma, dtype=torch.float32),
            )
            theta = proposal.sample_lhs(args.n_sims)   # (n, 2) [phi1, phi2]

        theta_np   = theta.numpy() if isinstance(theta, torch.Tensor) else theta
        theta_phys = degen_rotation.inverse_batch(theta_np)  # (n, 2) [Om, h]

        meta = {
            'proposal': proposal_name,
            'n_sims': args.n_sims,
            'rotated': True,
            'param_names': ['phi1', 'phi2'],
            'n1': degen_rotation.n1,
            'n2': degen_rotation.n2,
            'alpha': degen_rotation.alpha,
            'phi1_range': phi1_range,
            'phi2_range': phi2_range,
        }

    else:
        # ── original (Om, h) space ────────────────────────────────────────────
        a     = [OM_RANGE[0], H_RANGE[0]]
        b     = [OM_RANGE[1], H_RANGE[1]]
        sigma = [args.sigma_scale * (OM_RANGE[1] - OM_RANGE[0]),
                 args.sigma_scale * (H_RANGE[1]  - H_RANGE[0])]
        print(f'Om range: {OM_RANGE}')
        print(f'h  range: {H_RANGE}')

        loc   = [(a[i] + b[i]) / 2 for i in range(len(a))]
        scale = [(b[i] - a[i]) / 2 for i in range(len(a))]
        prior_assumed = ili.utils.IndependentNormal(loc=loc, scale=scale, device=device)

        if proposal_name == 'uniform':
            proposal = prior_assumed
            theta = sample_uniform_lhs(args.n_sims, [OM_RANGE, H_RANGE])
        else:
            proposal = cls_map[proposal_name](
                a=torch.tensor(a, dtype=torch.float32),
                b=torch.tensor(b, dtype=torch.float32),
                sigma=torch.tensor(sigma, dtype=torch.float32),
            )
            theta = proposal.sample_lhs(args.n_sims)   # (n, 2) [Om, h]

        theta_np   = theta.numpy() if isinstance(theta, torch.Tensor) else theta
        theta_phys = theta_np  # already physical

        meta = {
            'proposal': proposal_name,
            'n_sims': args.n_sims,
            'rotated': False,
            'param_names': ['Om', 'h'],
            'Om_range': OM_RANGE,
            'h_range':  H_RANGE,
        }

    # ── simulate ──────────────────────────────────────────────────────────────
    x = run_simulations(theta_phys)

    theta_t = torch.tensor(theta_np, dtype=torch.float32) if not isinstance(theta, torch.Tensor) else theta
    print(f'θ shape={theta_t.shape}  range=[{theta_t.min():.4f}, {theta_t.max():.4f}]')
    print(f'θ (Om,h) range: Om=[{theta_phys[:,0].min():.4f}, {theta_phys[:,0].max():.4f}]'
          f'  h=[{theta_phys[:,1].min():.4f}, {theta_phys[:,1].max():.4f}]')
    print(f'x shape={x.shape}')

    loader = NumpyLoader(x=x, theta=theta_t)

    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='maf', hidden_features=36, num_transforms=4),
    ]
    train_args = {'training_batch_size': 32, 'learning_rate': 1.34e-4}

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=prior_assumed, nets=nets, device=device,
        train_args=train_args, proposal=proposal,
        out_dir=out_dir,
    )
    posterior, summaries = runner(loader=loader)

    import json
    with open(os.path.join(out_dir, 'training_summaries.json'), 'w') as f:
        json.dump([{k: list(v) for k, v in s.items()} for s in summaries], f)

    np.save(os.path.join(out_dir, 'meta.npy'), meta)
    print(f'Done → {out_dir}/posterior.pkl')


if __name__ == '__main__':
    main()
