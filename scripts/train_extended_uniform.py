"""
Train one NPE model for the ExtendedUniform experiment.
Array index → proposal type:
  0  Uniform
  1  TailedUniform (σ = 0.1 × range_width, matched baseline)
  2  ExtendedUniform δ_scale=0.1
  3  ExtendedUniform δ_scale=0.3
  4  ExtendedUniform δ_scale=0.5
  5  ExtendedUniform δ_scale=1.0
"""
import argparse
import os
import sys
import random

import numpy as np
import torch

# Run from project root
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.imports import *
from toolbox.distributions import TailedUniform
from toolbox.simulators import sample_uniform_lhs

# ── Proposal catalogue ────────────────────────────────────────────────────────
PROPOSALS = [
    ('uniform',    None),
    ('tailed',     None),
    ('extended',   0.1),
    ('extended',   0.3),
    ('extended',   0.5),
    ('extended',   1.0),
]


def label_for(kind, delta_scale):
    if kind == 'uniform':
        return 'uniform'
    if kind == 'tailed':
        return 'taileduniform'
    return f'extended-uniform-delta-{delta_scale}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID (0–5)')
    parser.add_argument('--n_sims', type=int, default=6000)
    parser.add_argument('--out_root', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'toy-2-dim-models'))
    args = parser.parse_args()

    kind, delta_scale = PROPOSALS[args.task_id]
    run_label = label_for(kind, delta_scale)
    out_dir   = os.path.join(args.out_root, run_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] proposal={run_label}  out={out_dir}')

    # ── Reproducibility ───────────────────────────────────────────────────────
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Parameter space ───────────────────────────────────────────────────────
    param_ranges  = [(-1.0, 1.0), (-1.0, 1.0)]
    range_width   = 2.0
    sigma_scale   = 0.1
    sigma         = sigma_scale * range_width  # 0.2

    # ── Task ──────────────────────────────────────────────────────────────────
    task      = GaussianLinear(dim=2, prior_scale=range_width / 2)
    simulator = task.get_simulator()

    # Declared narrow prior (used by InferenceRunner for prior correction)
    prior_narrow = ili.utils.Uniform(
        low=[-1.0, -1.0], high=[1.0, 1.0], device=device
    )

    # ── Build proposal + sample θ ─────────────────────────────────────────────
    if kind == 'uniform':
        proposal = ili.utils.Uniform(
            low=[-1.0, -1.0], high=[1.0, 1.0], device=device
        )
        theta = sample_uniform_lhs(args.n_sims, param_ranges)

    elif kind == 'tailed':
        proposal = TailedUniform(
            a     = torch.tensor([-1.0, -1.0], dtype=torch.float32),
            b     = torch.tensor([ 1.0,  1.0], dtype=torch.float32),
            sigma = torch.tensor([sigma, sigma], dtype=torch.float32),
        )
        theta = proposal.sample_lhs(args.n_sims)

    else:  # extended uniform
        delta      = delta_scale * range_width
        ext_ranges = [(-1.0 - delta, 1.0 + delta)] * 2
        proposal   = ili.utils.Uniform(
            low =[ext_ranges[0][0], ext_ranges[1][0]],
            high=[ext_ranges[0][1], ext_ranges[1][1]],
            device=device,
        )
        theta = sample_uniform_lhs(args.n_sims, ext_ranges)

    print(f'θ shape={theta.shape}  range=[{theta.min():.3f}, {theta.max():.3f}]')

    # ── Simulate ──────────────────────────────────────────────────────────────
    x = simulator(theta)
    print(f'x shape={x.shape}')

    # ── Train NPE ─────────────────────────────────────────────────────────────
    loader = NumpyLoader(x=x, theta=theta)

    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='maf',  hidden_features=16, num_transforms=5),
        ili.utils.load_nde_sbi(engine='NPE', model='made', hidden_features=16, num_transforms=5),
    ]
    train_args = {'training_batch_size': 64, 'learning_rate': 5e-5}

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=prior_narrow, nets=nets, device=device,
        train_args=train_args, proposal=proposal,
        out_dir=out_dir,
    )
    posterior, summaries = runner(loader=loader)

    # ── Save loss curves ──────────────────────────────────────────────────────
    import json
    summary_path = os.path.join(out_dir, 'training_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump([{k: list(v) for k, v in s.items()} for s in summaries], f)

    print(f'Done. Posterior saved to {out_dir}/posterior.pkl')


if __name__ == '__main__':
    main()
