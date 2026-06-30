"""
Optuna hyperparameter search for NPE on the 2D cosmology (Omega_m, h) experiment.

Mirrors train_cosmo_2d.py in structure (SBI backend, IndependentNormal assumed prior,
same proposals), but searches over architecture hyperparameters.

One net per trial (model type is a hyperparameter: nsf or maf).
Data is regenerated fresh each trial from the specified proposal.
Evaluation uses a fixed test set generated once before the study begins.

Output: results/science/original/optuna/{proposal}/
  optuna_study.db          — SQLite storage (resumable with --n_trials)
  best_trial.yaml          — best trial number, value, and params
  nets/net-{i}/            — per-trial checkpoint, config, timing, and loss curve

Usage
-----
python scripts/run_cosmo_optuna.py --task_id 0
sbatch scripts/run_cosmo_optuna.sh
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
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

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

SIGMA_SCALE = 0.1
OM_RANGE = (0.27, 0.37)
H_RANGE  = (0.63, 0.71)

_cls_map = {
    'gaussiantailed':    GaussianTailed,
    'lineartailed':      LinearTailed,
    'exponentialtailed': ExponentialTailed,
    'uniformtailed':     UniformTailed,
}


def run_simulations(theta_phys_np, desc='Simulating'):
    xs = []
    for row in tqdm(theta_phys_np, desc=desc, leave=False):
        xs.append(syren_simulator(row))
    return np.array(xs, dtype=np.float32)


def build_prior_and_proposal(proposal_name, device):
    a = [OM_RANGE[0], H_RANGE[0]]
    b = [OM_RANGE[1], H_RANGE[1]]
    loc   = [(a[i] + b[i]) / 2 for i in range(2)]
    scale = [(b[i] - a[i]) / 2 for i in range(2)]
    prior_assumed = ili.utils.IndependentNormal(loc=loc, scale=scale, device=device)

    if proposal_name == 'uniform':
        proposal = prior_assumed
    else:
        sigma = [SIGMA_SCALE * (b[i] - a[i]) for i in range(2)]
        proposal = _cls_map[proposal_name](
            a=torch.tensor(a, dtype=torch.float32),
            b=torch.tensor(b, dtype=torch.float32),
            sigma=torch.tensor(sigma, dtype=torch.float32),
        )
    return prior_assumed, proposal


def generate_dataset(proposal_name, proposal, n_sims):
    if proposal_name == 'uniform':
        theta = sample_uniform_lhs(n_sims, [OM_RANGE, H_RANGE])
    else:
        theta = proposal.sample_lhs(n_sims)

    theta_np = theta.numpy() if isinstance(theta, torch.Tensor) else theta
    x = run_simulations(theta_np)
    theta_t = torch.tensor(theta_np, dtype=torch.float32)
    return x, theta_t


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
    f, ax = plt.subplots(figsize=(6, 4))
    for i, s in enumerate(summaries):
        ax.plot(s['validation_log_probs'], label=f'Net {i}', lw=1)
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(os.path.join(out_dir, 'loss.jpg'), dpi=100, bbox_inches='tight')
    plt.close(f)


def objective(
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help=f'0-{len(PROPOSALS)-1}: {PROPOSALS}')
    parser.add_argument('--n_sims', type=int, default=3000,
                        help='Training simulations per trial')
    parser.add_argument('--n_test_sims', type=int, default=500,
                        help='Fixed test set size (generated once at study start)')
    parser.add_argument('--n_trials', type=int, default=150,
                        help='Optuna trials per job')
    parser.add_argument('--out_root', type=str, default=None)
    args = parser.parse_args()

    if not (0 <= args.task_id < len(PROPOSALS)):
        raise ValueError(f'task_id must be 0-{len(PROPOSALS)-1}, got {args.task_id}')

    proposal_name = PROPOSALS[args.task_id]

    if args.out_root is None:
        args.out_root = os.path.join(ROOT, 'results', 'science', 'original', 'optuna')
    out_dir = os.path.join(args.out_root, proposal_name)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] proposal={proposal_name}'
          f'  n_sims={args.n_sims}  n_test={args.n_test_sims}'
          f'  n_trials={args.n_trials}  out={out_dir}')

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    prior_assumed, proposal = build_prior_and_proposal(proposal_name, device)

    def load_or_generate(x_path, theta_path, n_sims, label):
        if os.path.exists(x_path) and os.path.exists(theta_path):
            x     = np.load(x_path)
            theta = torch.tensor(np.load(theta_path), dtype=torch.float32)
            print(f'Loaded existing {label} from {out_dir}')
        else:
            print(f'Generating {label}...')
            x, theta = generate_dataset(proposal_name, proposal, n_sims)
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
        lambda trial: objective(
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


if __name__ == '__main__':
    main()
