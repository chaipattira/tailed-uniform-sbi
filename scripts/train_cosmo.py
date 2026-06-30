"""
Train NPE model(s) for the 2D cosmology (Omega_m, h) science experiment.

Two operating modes controlled by --optuna:

  Default (no --optuna)
  ---------------------
  Trains one NPE per proposal with fixed MAF architecture
  (hidden_features=36, num_transforms=4, batch_size=32, lr=1.34e-4).
  Supports --rotated / --no-rotated to work in (phi1, phi2) or (Om, h) space.

  Output: results/science/{rotated|original}/{proposal}/
    posterior.pkl          — trained posterior
    training_summaries.json
    meta.npy

  With --optuna
  -------------
  Runs an Optuna hyperparameter search over architecture and training
  hyperparameters (model type, hidden_features, num_transforms, batch_size,
  learning_rate).  Always operates in the ORIGINAL (Om, h) space (matching the
  original run_cosmo_optuna.py behaviour; --rotated / --no-rotated are ignored).
  Data is persisted across trials so the study is resumable.

  Output: results/science/original/optuna/{proposal}/
    optuna_study.db        — SQLite storage (resumable with --n_trials)
    best_trial.yaml        — best trial number, value, and params
    x_train.npy, theta_train.npy
    x_test.npy,  theta_test.npy
    nets/net-{i}/          — per-trial checkpoint, config, timing, and loss curve

Proposals (task_id):
  0: uniform           — flat uniform
  1: gaussiantailed    — Gaussian-tailed
  2: lineartailed      — linear-tailed
  3: exponentialtailed — exponential-tailed
  4: uniformtailed     — uniform-tailed

Usage
-----
# Default mode (fixed MAF, rotated space):
python scripts/train_cosmo.py --task_id 0 --rotated
sbatch scripts/train_cosmo.sh                     # array 0-4, rotated
sbatch scripts/train_cosmo.sh --no-rotated        # array 0-4, original

# Optuna mode:
python scripts/train_cosmo.py --task_id 0 --optuna
sbatch scripts/train_cosmo.sh --optuna            # array 0-4
"""
import argparse
import os
import random
import sys
import time
import yaml

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

PROPOSALS = [
    'uniform',
    'gaussiantailed',
    'lineartailed',
    'exponentialtailed',
    'uniformtailed',
]

SIGMA_SCALE = 0.1       # tail width as fraction of prior range
OM_RANGE = (0.27, 0.37) # ±3σ support
H_RANGE  = (0.63, 0.71) # ±3σ support

_cls_map = {
    'gaussiantailed':    GaussianTailed,
    'lineartailed':      LinearTailed,
    'exponentialtailed': ExponentialTailed,
    'uniformtailed':     UniformTailed,
}


# ── shared helpers ─────────────────────────────────────────────────────────────

def run_simulations(theta_phys_np, desc='Simulating'):
    """Run syren_simulator over a batch of (Om, h) parameters.
    theta_phys_np : ndarray (n, 2) — columns are [Om, h]
    """
    xs = []
    for row in tqdm(theta_phys_np, desc=desc, leave=False):
        xs.append(syren_simulator(row))
    return np.array(xs, dtype=np.float32)   # (n, n_k)


def build_prior_and_proposal_original(proposal_name, device, sigma_scale):
    """Build prior_assumed and proposal in original (Om, h) space."""
    a = [OM_RANGE[0], H_RANGE[0]]
    b = [OM_RANGE[1], H_RANGE[1]]
    loc   = [(a[i] + b[i]) / 2 for i in range(2)]
    scale = [(b[i] - a[i]) / 2 for i in range(2)]
    prior_assumed = ili.utils.IndependentNormal(loc=loc, scale=scale, device=device)

    if proposal_name == 'uniform':
        proposal = prior_assumed
    else:
        sigma = [sigma_scale * (b[i] - a[i]) for i in range(2)]
        proposal = _cls_map[proposal_name](
            a=torch.tensor(a, dtype=torch.float32),
            b=torch.tensor(b, dtype=torch.float32),
            sigma=torch.tensor(sigma, dtype=torch.float32),
        )
    return prior_assumed, proposal, a, b


def generate_dataset_original(proposal_name, proposal, n_sims):
    """Sample from proposal in original space and simulate."""
    if proposal_name == 'uniform':
        theta = sample_uniform_lhs(n_sims, [OM_RANGE, H_RANGE])
    else:
        theta = proposal.sample_lhs(n_sims)

    theta_np = theta.numpy() if isinstance(theta, torch.Tensor) else theta
    x = run_simulations(theta_np)
    theta_t = torch.tensor(theta_np, dtype=torch.float32)
    return x, theta_t


# ── default mode ───────────────────────────────────────────────────────────────

def run_default(args):
    """
    Train one NPE with fixed MAF (hidden_features=36, num_transforms=4).
    Supports both rotated (phi1, phi2) and original (Om, h) spaces.
    Output: results/science/{rotated|original}/{proposal}/
    """
    from toolbox.priors import get_rotated_priors
    from toolbox.transforms import degen_rotation
    import json

    coord_subdir = 'rotated' if args.rotated else 'original'
    if args.out_root is None:
        out_root = os.path.join(ROOT, 'results', 'science', coord_subdir)
    else:
        out_root = args.out_root

    proposal_name = PROPOSALS[args.task_id]
    out_dir = os.path.join(out_root, proposal_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] proposal={proposal_name}  rotated={args.rotated}'
          f'  sigma_scale={args.sigma_scale}  n_sims={args.n_sims}  out={out_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    if args.rotated:
        # ── rotated (phi1, phi2) space ─────────────────────────────────────────
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
            proposal = _cls_map[proposal_name](
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
        # ── original (Om, h) space ─────────────────────────────────────────────
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
            proposal = _cls_map[proposal_name](
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

    # ── simulate ───────────────────────────────────────────────────────────────
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

    with open(os.path.join(out_dir, 'training_summaries.json'), 'w') as f:
        json.dump([{k: list(v) for k, v in s.items()} for s in summaries], f)

    np.save(os.path.join(out_dir, 'meta.npy'), meta)
    print(f'Done → {out_dir}/posterior.pkl')


# ── optuna mode ────────────────────────────────────────────────────────────────

def evaluate_posterior(posterior, x_test, theta_test):
    """Mean unnormalized log_prob over (x_i, theta_i) test pairs."""
    x_t  = torch.tensor(x_test, dtype=torch.float32) if not isinstance(x_test, torch.Tensor) else x_test
    th_t = theta_test if isinstance(theta_test, torch.Tensor) else torch.tensor(theta_test, dtype=torch.float32)
    log_probs = []
    for i in range(len(x_t)):
        lp = posterior.log_prob(
            theta=th_t[i:i+1],
            x=x_t[i:i+1],
            norm_posterior=False,
        )
        log_probs.append(lp.item())
    return float(np.mean(log_probs))


def plot_training_history(summaries, out_dir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(summaries):
        ax.plot(s['validation_log_probs'], label=f'Net {i}', lw=1)
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(os.path.join(out_dir, 'loss.jpg'), dpi=100, bbox_inches='tight')
    plt.close(f)


def optuna_objective(
    trial,
    loader,
    x_test, theta_test,
    prior_assumed, proposal,
    out_dir, device,
):
    trial_num = trial.number
    exp_dir = os.path.join(out_dir, 'nets', f'net-{trial_num}')
    os.makedirs(exp_dir, exist_ok=True)

    model           = trial.suggest_categorical('model', ['nsf', 'maf'])
    hidden_features = trial.suggest_int('hidden_features', 4, 64, log=True)
    num_transforms  = trial.suggest_int('num_transforms', 2, 4)
    batch_size      = int(2 ** trial.suggest_int('log2_batch_size', 3, 5))
    learning_rate   = trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True)

    mcfg = dict(
        model=model,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        batch_size=batch_size,
        learning_rate=learning_rate,
    )
    print('~' * 20 + f' Trial {trial_num} ' + '~' * 20)
    for k, v in mcfg.items():
        print(f'\t{k}: {v}')

    nets = [ili.utils.load_nde_sbi(
        engine='NPE', model=model,
        hidden_features=hidden_features, num_transforms=num_transforms,
    )]
    train_args = {'training_batch_size': batch_size, 'learning_rate': learning_rate}

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=prior_assumed, nets=nets, device=device,
        train_args=train_args, proposal=proposal,
        out_dir=exp_dir,
    )

    t0 = time.time()
    posterior, summaries = runner(loader=loader)
    elapsed = time.time() - t0

    with open(os.path.join(exp_dir, 'timing.txt'), 'w') as f:
        f.write(f'{elapsed:.3f}')
    with open(os.path.join(exp_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(mcfg, f)
    plot_training_history(summaries, exp_dir)

    log_prob_test = evaluate_posterior(posterior, x_test, theta_test)
    with open(os.path.join(exp_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{log_prob_test}\n')

    print(f'Trial {trial_num}: log_prob_test = {log_prob_test:.4f}  ({elapsed:.0f}s)')
    return log_prob_test


def run_optuna(args):
    """
    Run an Optuna hyperparameter search in original (Om, h) space.
    Output: results/science/original/optuna/{proposal}/
    """
    import optuna

    proposal_name = PROPOSALS[args.task_id]

    if args.out_root is None:
        out_root = os.path.join(ROOT, 'results', 'science', 'original', 'optuna')
    else:
        out_root = args.out_root
    out_dir = os.path.join(out_root, proposal_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] proposal={proposal_name}'
          f'  n_sims={args.n_sims}  n_test={args.n_test_sims}'
          f'  n_trials={args.n_trials}  out={out_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    prior_assumed, proposal, _, _ = build_prior_and_proposal_original(
        proposal_name, device, args.sigma_scale
    )

    def load_or_generate(x_path, theta_path, n_sims, label):
        if os.path.exists(x_path) and os.path.exists(theta_path):
            x     = np.load(x_path)
            theta = torch.tensor(np.load(theta_path), dtype=torch.float32)
            print(f'Loaded existing {label} from {out_dir}')
        else:
            print(f'Generating {label}...')
            x, theta = generate_dataset_original(proposal_name, proposal, n_sims)
            np.save(x_path, x)
            np.save(theta_path, theta.numpy())
        print(f'{label}: x={x.shape}  theta={theta.shape}')
        return x, theta

    x_train, theta_train = load_or_generate(
        os.path.join(out_dir, 'x_train.npy'),
        os.path.join(out_dir, 'theta_train.npy'),
        args.n_sims, 'training set',
    )
    x_test, theta_test = load_or_generate(
        os.path.join(out_dir, 'x_test.npy'),
        os.path.join(out_dir, 'theta_test.npy'),
        args.n_test_sims, 'test set',
    )

    loader = NumpyLoader(x=x_train, theta=theta_train)

    study = optuna.create_study(
        study_name=f'cosmo_{proposal_name}',
        direction='maximize',
        storage=f'sqlite:///{os.path.join(out_dir, "optuna_study.db")}',
        load_if_exists=True,
    )
    study.optimize(
        lambda trial: optuna_objective(
            trial,
            loader,
            x_test, theta_test,
            prior_assumed, proposal,
            out_dir, device,
        ),
        n_trials=args.n_trials,
        n_jobs=1,
        timeout=60 * 60 * 11.5,
        show_progress_bar=False,
        gc_after_trial=True,
    )

    best = study.best_trial
    print(f'Best trial: {best.number}  log_prob_test: {best.value:.4f}')
    print(f'Best params: {best.params}')
    with open(os.path.join(out_dir, 'best_trial.yaml'), 'w') as f:
        yaml.dump({'trial': best.number, 'value': float(best.value), 'params': best.params}, f)


# ── entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Train NPE for 2D cosmology. Use --optuna for hyperparameter search.'
    )
    parser.add_argument('--task_id', type=int, required=True,
                        help=f'0-{len(PROPOSALS)-1}: {PROPOSALS}')
    parser.add_argument('--n_sims', type=int, default=3000,
                        help='Training simulations (default: %(default)s)')
    parser.add_argument('--out_root', type=str, default=None,
                        help='Override output root directory')
    parser.add_argument('--sigma_scale', type=float, default=SIGMA_SCALE,
                        help='Tail width as fraction of prior range (default: %(default)s)')
    # default-mode only
    parser.add_argument('--rotated', action=argparse.BooleanOptionalAction, default=True,
                        help='[default mode] Train in rotated (phi1,phi2) space (default) or original (Om,h) space')
    # optuna-mode flag and its args
    parser.add_argument('--optuna', action='store_true', default=False,
                        help='Run Optuna hyperparameter search instead of fixed-architecture training')
    parser.add_argument('--n_trials', type=int, default=150,
                        help='[--optuna] Number of Optuna trials (default: %(default)s)')
    parser.add_argument('--n_test_sims', type=int, default=500,
                        help='[--optuna] Fixed test set size (default: %(default)s)')
    args = parser.parse_args()

    if not (0 <= args.task_id < len(PROPOSALS)):
        raise ValueError(f'task_id must be 0-{len(PROPOSALS)-1}, got {args.task_id}')

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    if args.optuna:
        run_optuna(args)
    else:
        run_default(args)


if __name__ == '__main__':
    main()
