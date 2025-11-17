import time
import yaml
import torch
import ili
import optuna
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
from scipy.stats import qmc
import math
from torch.distributions import Distribution, Uniform, HalfNormal
from torch.distributions.utils import broadcast_all
from ili.utils.distributions_pt import CustomIndependent


class TailedNormal(Distribution):
    arg_constraints = {
        'a': torch.distributions.constraints.real,
        'b': torch.distributions.constraints.dependent,
        'sigma': torch.distributions.constraints.positive,
    }
    support = torch.distributions.constraints.real
    has_rsample = False

    def __init__(self, a, b, sigma, validate_args=None):
        self.a, self.b, self.sigma = broadcast_all(a, b, sigma)
        if torch.any(self.a >= self.b):
            raise ValueError("`a` must be less than `b`.")

        self.Z = math.sqrt(2 * math.pi) * self.sigma + (self.b - self.a)
        self.A = math.sqrt(2 * math.pi) * self.sigma / self.Z
        self.B = (self.b - self.a) / self.Z

        self.halfnormal = HalfNormal(self.sigma)

        super().__init__(batch_shape=self.a.size(), validate_args=validate_args)

    def log_prob(self, x):
        x, a, b, sigma = broadcast_all(x, self.a, self.b, self.sigma)

        logA = torch.log(self.A.to(dtype=x.dtype, device=x.device))
        logB = torch.log(self.B.to(dtype=x.dtype, device=x.device))
        log_uniform = logB - torch.log(b - a)

        # left: x <= a => z = a - x
        z_left = torch.abs(a - x)
        log_halfnorm_left = self.halfnormal.log_prob(z_left) + logA - math.log(2.0)

        # right: x >= b => z = x - b
        z_right = torch.abs(x - b)
        log_halfnorm_right = self.halfnormal.log_prob(z_right) + logA - math.log(2.0)

        return torch.where(x <= a, log_halfnorm_left,
               torch.where(x >= b, log_halfnorm_right,
               log_uniform))

    def cdf(self, x):
        x, a, b, sigma = broadcast_all(x, self.a, self.b, self.sigma)
        sqrt2 = math.sqrt(2.0)

        def Phi(z):  # Standard Normal CDF
            return 0.5 * (1 + torch.erf(z / sqrt2))

        left_cdf = self.A * Phi((x - a) / sigma)
        center_cdf = 0.5 * self.A + self.B * (x - a) / (b - a)
        right_cdf = self.B + self.A * Phi((x - b) / sigma)

        return torch.where(x <= a, left_cdf,
               torch.where(x >= b, right_cdf,
               center_cdf))

    def icdf(self, u):
        # Helper function for the Inverse Standard Normal CDF
        def inv_Phi(p):
            # Clamping p to avoid NaNs from erfinv at the boundaries 0 and 1
            p_clamped = torch.clamp(p, 1e-9, 1.0 - 1e-9)
            return math.sqrt(2.0) * torch.erfinv(2.0 * p_clamped - 1.0)

        # Thresholds dividing the distribution regions
        thresh_left = 0.5 * self.A
        thresh_right = 1.0 - 0.5 * self.A

        u_left_norm = u / self.A
        left_tail = self.a + self.sigma * inv_Phi(u_left_norm)

        u_right_norm = (u - self.B) / self.A
        right_tail = self.b + self.sigma * inv_Phi(u_right_norm)

        u_middle_norm = (u - thresh_left) / self.B
        middle = self.a + u_middle_norm * (self.b - self.a)

        return torch.where(u < thresh_left, left_tail,
                           torch.where(u > thresh_right, right_tail, middle))


    def sample(self, sample_shape=torch.Size()):
        u = torch.rand(sample_shape + self.a.shape, device=self.a.device)
        thresh_left = 0.5 * self.A
        thresh_right = 1.0 - 0.5 * self.A

        left_tail = self.a - self.halfnormal.sample(sample_shape)
        right_tail = self.b + self.halfnormal.sample(sample_shape)

        x_middle = self.a + (u - thresh_left) * (self.b - self.a) / self.B

        return torch.where(u < thresh_left, left_tail,
               torch.where(u > thresh_right, right_tail,
               x_middle))

    def sample_lhs(self, n_samples):
        """Sample using Latin Hypercube Sampling"""
        # Generate LHS samples in [0,1]^d
        sampler = qmc.LatinHypercube(d=len(self.a.flatten()), seed=42)
        u_samples = sampler.random(n_samples)
        u_tensor = torch.tensor(u_samples, dtype=torch.float32, device=self.a.device)

        # Transform using inverse CDF
        return self.icdf(u_tensor)

    def mean(self):
        return 0.5 * (self.a + self.b)


def get_priors_and_proposals(device='cpu'):
    """
    Get priors and proposals for toy problem
    """
    # Parameters ranges
    param_1_range = (-1.0, 1.0)
    param_2_range = (-1.0, 1.0)
    param_ranges = [param_1_range, param_2_range]

    param_1_width = param_1_range[1] - param_1_range[0]
    param_2_width = param_2_range[1] - param_2_range[0]

    # Scale sigma relative to parameter ranges
    sigma_scale = 0.1
    sigmas = [sigma_scale * (high - low) for low, high in param_ranges]

    # Prior
    prior = ili.utils.IndependentNormal(
        loc=[param_1_width/2, param_2_width/2],
        scale=sigmas,
        device=device
    )

    # Uniform proposal
    proposal_uniform = ili.utils.Uniform(
        low=[param_1_range[0], param_2_range[0]],
        high=[param_1_range[1], param_2_range[1]],
        device=device
    )

    # TailedNormal proposal
    proposal_tailed = TailedNormal(
        a=torch.tensor([param_1_range[0], param_2_range[0]], dtype=torch.float32),
        b=torch.tensor([param_1_range[1], param_2_range[1]], dtype=torch.float32),
        sigma=torch.tensor([sigmas[0], sigmas[1]], dtype=torch.float32),
    )

    return prior, proposal_uniform, proposal_tailed


def get_hyperprior():
    hyperprior = dict(
        model=['nsf', 'maf'],
        hidden_features=(4, 64),
        num_transforms=(1, 5),
        log2_batch_size=(3, 8),
        learning_rate=(1e-4, 1e-2),
    )
    return hyperprior


def evaluate_posterior(posterior, x, theta):
    log_prob = posterior.log_prob(theta=theta, x=x)
    return log_prob.mean()


def plot_training_history(histories, out_dir):
    # Plot training history
    f, ax = plt.subplots(1, 1, figsize=(6, 4))
    for i, h in enumerate(histories):
        ax.plot(h['validation_log_probs'], label=f'Net {i}', lw=1)
    ax.set(xlabel='Epoch', ylabel='Validation log prob')
    ax.legend()
    f.savefig(join(out_dir, 'loss.jpg'), dpi=100, bbox_inches='tight')
    plt.close(f)


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
    log_prob_test = evaluate_posterior(posterior, x_test, theta_test)
    with open(join(exp_dir, 'log_prob_test.txt'), 'w') as f:
        f.write(f'{log_prob_test}\n')

    # plot training history
    plot_training_history(summaries, exp_dir)

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

    # get device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # get priors and proposals
    prior, proposal_uniform, proposal_tailed = get_priors_and_proposals(device)
    if 'tailed' in model_name:
        proposal = proposal_tailed
    elif 'uniform' in model_name:
        proposal = proposal_uniform
    else:
        raise ValueError('Model name must contain "tailed" or "uniform"')

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
    model_library = '/home/x-ctirapongpra/scratch/tailed-uniform-sbi/toy-2-dim-models'
    model_name = 'uniform_toy'  # 'tailed_toy' or 'uniform_toy'
    Nexp_per_job = 50  # number of experiments per SLURM job

    run_experiment(model_library, model_name, Nexp_per_job)
