"""
Train one NPE model for the inference-dim-waste experiment.
Array index → (dim, proposal):
  dim_idx      = task_id // 4   (0→4, 1→8, 2→16)
  proposal_idx = task_id  % 4   (0→uniform, 1→tailed, 2→extended 0.1, 3→extended 0.3)
"""
import argparse
import os
import sys
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.imports import *
from toolbox.distributions import TailedUniform
from toolbox.simulators import sample_uniform_lhs

DIMS = [4, 8, 16]
PROPOSALS = [
    ('uniform',   None),
    ('tailed',    None),
    ('extended',  0.1),
    ('extended',  0.3),
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
                        help='SLURM_ARRAY_TASK_ID (0-11)')
    parser.add_argument('--n_sims', type=int, default=6000)
    parser.add_argument('--out_root', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'inference-dim-models'))
    args = parser.parse_args()

    if not (0 <= args.task_id <= 11):
        raise ValueError(f'task_id must be 0-11, got {args.task_id}')

    dim_idx      = args.task_id // 4
    proposal_idx = args.task_id % 4
    dim          = DIMS[dim_idx]
    kind, delta_scale = PROPOSALS[proposal_idx]
    run_label    = label_for(kind, delta_scale)
    out_dir      = os.path.join(args.out_root, f'dim{dim}', run_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] dim={dim}  proposal={run_label}  out={out_dir}')

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    param_ranges = [(-1.0, 1.0)] * dim
    range_width  = 2.0
    sigma        = 0.1 * range_width  # 0.2, per-dimension for TailedUniform

    task      = GaussianLinear(dim=dim, prior_scale=range_width / 2)
    simulator = task.get_simulator()

    prior_narrow = ili.utils.Uniform(
        low =[-1.0] * dim,
        high=[ 1.0] * dim,
        device=device,
    )

    if kind == 'uniform':
        proposal = ili.utils.Uniform(
            low =[-1.0] * dim,
            high=[ 1.0] * dim,
            device=device,
        )
        theta = sample_uniform_lhs(args.n_sims, param_ranges)

    elif kind == 'tailed':
        proposal = TailedUniform(
            a     = torch.tensor([-1.0] * dim, dtype=torch.float32),
            b     = torch.tensor([ 1.0] * dim, dtype=torch.float32),
            sigma = torch.tensor([sigma]  * dim, dtype=torch.float32),
        )
        theta = proposal.sample_lhs(args.n_sims)

    else:  # extended uniform
        delta      = delta_scale * range_width
        ext_low    = -1.0 - delta
        ext_high   =  1.0 + delta
        ext_ranges = [(ext_low, ext_high)] * dim
        proposal   = ili.utils.Uniform(
            low =[ext_low]  * dim,
            high=[ext_high] * dim,
            device=device,
        )
        theta = sample_uniform_lhs(args.n_sims, ext_ranges)

    print(f'θ shape={theta.shape}  range=[{theta.min():.3f}, {theta.max():.3f}]')

    x = simulator(theta)
    print(f'x shape={x.shape}')

    loader = NumpyLoader(x=x, theta=theta)

    # hidden_features=50 (vs 16 in 2D baseline) gives adequate capacity at dim=16;
    # kept constant across all dims so only proposal type varies between runs.
    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='maf',  hidden_features=50, num_transforms=5),
        ili.utils.load_nde_sbi(engine='NPE', model='made', hidden_features=50, num_transforms=5),
    ]
    train_args = {'training_batch_size': 64, 'learning_rate': 5e-5}

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=prior_narrow, nets=nets, device=device,
        train_args=train_args, proposal=proposal,
        out_dir=out_dir,
    )
    posterior, summaries = runner(loader=loader)

    import json
    summary_path = os.path.join(out_dir, 'training_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump([{k: list(v) for k, v in s.items()} for s in summaries], f)

    print(f'Done. Posterior saved to {out_dir}/posterior.pkl')


if __name__ == '__main__':
    main()
