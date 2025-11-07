
import time
import yaml
import torch
import ili
import optuna
from os.path import join
import numpy as np
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
from toolbox.priors import get_priors


def get_hyperprior():
    hyperprior = dict(
        model=['nsf'],
        hidden_features=(4, 64),
        num_transforms=(1, 5),
        log2_batch_size=(3, 8),
        learning_rate=(1e-4, 1e-2),
    )
    return hyperprior


def evaluate_posterior(posterior, x, theta):
    log_prob = posterior.log_prob(theta=theta, x=x)
    return log_prob.mean()


def objective(
    trial,
    x_train, theta_train,
    x_test, theta_test,
    prior, proposal,
    model_dir, device='cpu'
):

    trial_num = trial.number
    exp_dir = join(model_dir, 'nets', f'net-{trial_num}')

    # Sample hyperparameters
    hyperprior = get_hyperprior()
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
    print('~'*20 + f' Trial {trial_num} ' + '~'*20)
    print('CONFIGURATION :')
    for k, v in mcfg.items():
        print(f'\t{k}: {v}')

    # setup loader
    loader = NumpyLoader(x=x_train, theta=theta_train)

    # setup networks and hyperparameters
    nets = [ili.utils.load_nde_lampe(
        engine='NPE', model=model,
        hidden_features=hidden_features, num_transforms=num_transforms
    )]
    train_args = {
        'training_batch_size': batch_size,
        'learning_rate': learning_rate
    }
    runner = InferenceRunner.load(
        backend='lampe',
        engine='NPE',
        prior=prior,
        nets=nets,
        device=None,
        train_args=train_args,
        proposal=proposal,
        out_dir=exp_dir
    )

    start = time.time()
    posterior, summaries = runner(loader=loader)
    end = time.time()

    # Save the timing and metadata
    with open(join(exp_dir, 'timing.txt'), 'w') as f:
        f.write(f'{end - start:.3f}')
    with open(join(exp_dir, 'model_config.yaml'), 'w') as f:
        yaml.dump(mcfg, f)

    # evaluate the posterior and save to file
    log_prob_test = evaluate_posterior(
        posterior, x_test, theta_test)
    log_prob_test = summaries[0]['best_validation_log_prob']
    with open(join(exp_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{log_prob_test}\n')

    return log_prob_test


def run_experiment(
    model_library, model_name,
    Nexp_per_job=50
):
    model_dir = join(model_library, model_name)

    # load data
    x_train = np.load(join(model_dir, 'x_train.npy'))
    theta_train = np.load(join(model_dir, 'theta_train.npy'))
    x_test = np.load(join(model_dir, 'x_test.npy'))
    theta_test = np.load(join(model_dir, 'theta_test.npy'))
    print(f'Train length = {len(x_train)} | Test length = {len(x_test)}')

    # get priors and proposals
    prior_normal, prior_uniform, prior_tailed = get_priors()
    if 'tailed' in model_name:
        prior = prior_normal
        proposal = prior_tailed
    elif 'uniform' in model_name:
        prior = prior_normal
        proposal = prior_uniform
    else:
        raise ValueError('Model name must contain "tailed" or "uniform"')

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # run hyperparameter optimization
    print('Running hyperparameter optimization...')
    study = optuna.load_study(
        study_name=model_name,
        storage=f'sqlite:///{join(model_dir, "optuna_study.db")}',
    )
    study.optimize(
        lambda trial: objective(
            trial,
            x_train, theta_train,
            x_test, theta_test,
            prior, proposal,
            model_dir, device=device),
        n_trials=Nexp_per_job,
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
    model_library = '../data/sci-2-dim-models'
    model_name = 'tailed_power'  # 'tailed_power' or 'uniform_power'

    Nexp_per_job = 20  # number of experiments per SLURM job

    run_experiment(model_library, model_name, Nexp_per_job)
