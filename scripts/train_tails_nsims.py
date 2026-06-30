"""
Train one NPE model for the tails-nsims experiment (Exp 2 — simulation budget sweep).
Array index → (dim, n_sims, proposal):
  n_per_dim    = len(N_SIMS_LIST) * len(PROPOSALS)   # 20
  dim_idx      = task_id // n_per_dim
  nsims_idx    = (task_id % n_per_dim) // len(PROPOSALS)
  proposal_idx = task_id % len(PROPOSALS)
"""
import argparse
import os
import sys
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from sbibm.tasks.gaussian_linear.task import GaussianLinear
from sbibm.tasks.gaussian_linear_uniform.task import GaussianLinearUniform
from toolbox.distributions import GaussianTailed, LinearTailed, ExponentialTailed, UniformTailed
from toolbox.simulators import sample_uniform_lhs

DIMS        = [4, 8]
N_SIMS_LIST = [500, 1000, 5000, 10000]
PROPOSALS   = ['uniform', 'gaussiantailed', 'lineartailed', 'exponentialtailed', 'uniformtailed']


def build_proposal(name, dim, sigma, device):
    a = torch.tensor([-1.0] * dim, dtype=torch.float32)
    b = torch.tensor([ 1.0] * dim, dtype=torch.float32)
    s = torch.tensor([sigma]  * dim, dtype=torch.float32)
    cls = {'gaussiantailed': GaussianTailed, 'lineartailed': LinearTailed,
           'exponentialtailed': ExponentialTailed, 'uniformtailed': UniformTailed}[name]
    return cls(a=a, b=b, sigma=s)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True)
    parser.add_argument('--out_root', type=str, default=None)
    parser.add_argument('--task_name', default='gaussianlinear',
                        choices=['gaussianlinear', 'gaussianlinearuniform'])
    args = parser.parse_args()
    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', args.task_name, 'exp2-nsims')

    n_per_dim = len(N_SIMS_LIST) * len(PROPOSALS)
    n_tasks   = len(DIMS) * n_per_dim
    if not (0 <= args.task_id < n_tasks):
        raise ValueError(f'task_id must be 0-{n_tasks - 1}, got {args.task_id}')

    dim_idx      = args.task_id // n_per_dim
    nsims_idx    = (args.task_id % n_per_dim) // len(PROPOSALS)
    proposal_idx = args.task_id % len(PROPOSALS)

    dim           = DIMS[dim_idx]
    n_sims        = N_SIMS_LIST[nsims_idx]
    proposal_name = PROPOSALS[proposal_idx]
    out_dir       = os.path.join(args.out_root, f'dim{dim}', f'n{n_sims}', proposal_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] dim={dim}  n_sims={n_sims}  proposal={proposal_name}  out={out_dir}')

    SEED = 42
    np.random.seed(SEED); torch.manual_seed(SEED); random.seed(SEED)

    device      = 'cuda' if torch.cuda.is_available() else 'cpu'
    range_width = 2.0
    sigma       = 0.1 * range_width
    param_ranges = [(-1.0, 1.0)] * dim

    if args.task_name == 'gaussianlinear':
        task         = GaussianLinear(dim=dim, prior_scale=range_width / 2)
        prior_assumed = ili.utils.IndependentNormal(loc=[0.0]*dim, scale=[range_width/2]*dim, device=device)
    else:
        task          = GaussianLinearUniform(dim=dim)
        prior_assumed = ili.utils.Uniform(low=[-1.0]*dim, high=[1.0]*dim, device=device)
    simulator = task.get_simulator()

    if proposal_name == 'uniform':
        proposal = ili.utils.Uniform(low=[-1.0]*dim, high=[1.0]*dim, device=device)
        theta    = sample_uniform_lhs(n_sims, param_ranges)
    else:
        proposal = build_proposal(proposal_name, dim, sigma, device)
        theta    = proposal.sample_lhs(n_sims)

    print(f'θ shape={theta.shape}  range=[{theta.min():.3f}, {theta.max():.3f}]')
    x = simulator(theta)
    print(f'x shape={x.shape}')

    loader = NumpyLoader(x=x, theta=theta)
    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='maf',  hidden_features=50, num_transforms=5),
        ili.utils.load_nde_sbi(engine='NPE', model='made', hidden_features=50, num_transforms=5),
    ]
    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=prior_assumed, nets=nets, device=device,
        train_args={'training_batch_size': 64, 'learning_rate': 5e-5},
        proposal=proposal, out_dir=out_dir,
    )
    posterior, summaries = runner(loader=loader)

    import json
    with open(os.path.join(out_dir, 'training_summaries.json'), 'w') as f:
        json.dump([{k: list(v) for k, v in s.items()} for s in summaries], f)

    print(f'Done. Posterior saved to {out_dir}/posterior.pkl')


if __name__ == '__main__':
    main()
