from toolbox.simulators import syren_simulator, sample_uniform_lhs
from toolbox.distributions import TailedNormal, IndependentTailedNormal
from ili.utils.distributions_pt import IndependentTruncatedNormal
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
import ili
import seaborn as sns
from scipy.stats import truncnorm
from os.path import join
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import optuna
import time
import yaml

hyperprior = dict(
    model=['nsf'],
    hidden_features=(4, 64),
    num_transforms=(1, 5),
    log2_batch_size=(3, 8),
    learning_rate=(1e-5, 1e-2),
)


def objective(trial, x, theta, model_dir):

    trial_num = trial.number
    exp_dir = join(model_dir, 'nets', f'net-{trial_num}')

    # Sample hyperparameters
    model = trial.suggest_categorical("model", hyperprior['model'])
    hidden_features = trial.suggest_int(
        "hidden_features", *hyperprior['hidden_features'], log=True)
    num_transforms = trial.suggest_int(
        "num_transforms", *hyperprior['num_transforms'])
    batch_size = int(2**trial.suggest_int(
        "log2_batch_size", *hyperprior['log2_batch_size']))
    learning_rate = trial.suggest_float(
        "learning_rate", *hyperprior['learning_rate'], log=True)
    mcfg = dict(
        model=model,
        hidden_features=hidden_features,
        num_transforms=num_transforms,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    print(f'CONFIGURATION : {mcfg}')

    # setup loader
    loader = NumpyLoader(x=x, theta=theta)

    # setup networks and hyperparameters
    nets = [ili.utils.load_nde_lampe(
        engine='NPE', model='maf',
        hidden_features=16, num_transforms=5
    )]
    train_args = {
        'training_batch_size': 64,
        'learning_rate': 5e-5
    }
    runner = InferenceRunner.load(
        backend='lampe',
        engine='NPE',
        prior=None,
        nets=nets,
        device=None,
        train_args=train_args,
        proposal=None,
        out_dir=exp_dir
    )

    start = time.time()
    posterior_ensemble, summaries = runner(loader=loader)
    end = time.time()

    # Save the timing and metadata
    with open(join(exp_dir, 'timing.txt'), 'w') as f:
        f.write(f'{end - start:.3f}')
    with open(join(exp_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(mcfg, f)

    # evaluate the posterior and save to file
    # log_prob_test = evaluate_posterior(
    #     posterior, x_test, theta_test)
    log_prob_test = summaries['log_prob_test']
    with open(join(exp_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{log_prob_test}\n')

    return log_prob_test


def run_experiment(model_library, model_name):
    model_dir = join(model_library, model_name)

    # load data
    x = np.load(join(model_dir, 'x.npy'))
    theta = np.load(join(model_dir, 'theta.npy'))

    print(f'dataset length = {len(x)}')

    # run hyperparameter optimization
    print('Running hyperparameter optimization...')
    study = optuna.load_study(
        study_name=model_name,
        storage=f'sqlite:///{join(model_dir, "optuna_study.db")}',
    )
    study.optimize(
        lambda trial: objective(trial, x, theta, model_dir),
        n_trials=100,
        n_jobs=1,
        timeout=60*60*4,  # 4 hours
        show_progress_bar=False,
        gc_after_trial=True
    )
    # NOTE: n_jobs>1 doesn't seem to speed things up much,
    # It seems processes are fighting for threads.
    # Instead, we parallelize via SLURM


if __name__ == '__main__':
    # experiment settings
    model_library = '/Users/maho/git/tailed-normal-sbi/notebooks/sci-2-dim-models'
    model_name = 'tailed_power'

    run_experiment(model_library, model_name)
