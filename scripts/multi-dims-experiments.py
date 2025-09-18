import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from matplotlib.cm import get_cmap
from scipy.stats import truncnorm, qmc
from tqdm import tqdm
import sbibm
from sbibm.tasks.gaussian_linear.task import GaussianLinear
import scipy.stats as stats
from scipy.stats import wasserstein_distance, ks_2samp
import seaborn as sns
from itertools import product
import random
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split
import scipy.stats as stats
from scipy.stats import uniform
from matplotlib import colormaps
import matplotlib.colors as mcolors
import ili
from ili.dataloaders import NumpyLoader
from ili.inference import InferenceRunner
from ili.validation.metrics import PosteriorCoverage, PlotSinglePosterior
from torch.distributions import Distribution, Uniform, HalfNormal
from torch.distributions.utils import broadcast_all
import math
from ili.utils.distributions_pt import CustomIndependent
import itertools
import pickle

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)

dimensions = [8, 16]
n_posterior_samples = 1000

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
        sampler = qmc.LatinHypercube(d=len(self.a.flatten()), seed=13)
        u_samples = sampler.random(n_samples)
        u_tensor = torch.tensor(u_samples, dtype=torch.float32, device=self.a.device)
        
        # Transform using inverse CDF
        return self.icdf(u_tensor)

    def mean(self):
        return 0.5 * (self.a + self.b)

IndependentTailedNormal = type('IndependentTailedNormal', (CustomIndependent,), {'Distribution': TailedNormal})


class CircleEvaluator:
    def __init__(self, simulator, param_ranges, task):
        self.simulator = simulator
        self.param_ranges = param_ranges
        self.task = task
        self.dim = len(param_ranges)
        self.prior_center = np.array([(low + high) / 2 for low, high in param_ranges])
        self.max_radius = min([(high - low) / 2 for low, high in param_ranges])

    def create_test_points(self, n_radii=10, n_angles_per_dim=10):
        """Create test points on concentric hyperspheres"""
        test_points = [self.prior_center.copy()]
        radii = [0.0]

        for radius in np.linspace(0.2 * self.max_radius, 0.9 * self.max_radius, n_radii):
            # Generate points on unit hypersphere then scale by radius
            n_sphere_points = n_angles_per_dim ** (self.dim - 1) if self.dim > 2 else n_angles_per_dim * 6
            
            # Sample from unit sphere using Gaussian method
            for _ in range(n_sphere_points):
                # Generate random point on unit sphere
                point = np.random.randn(self.dim)
                point = point / np.linalg.norm(point)
                
                # Scale by radius and translate to center
                scaled_point = self.prior_center + radius * point
                
                # Check if point is within bounds
                if all(self.param_ranges[i][0] <= scaled_point[i] <= self.param_ranges[i][1] 
                       for i in range(self.dim)):
                    test_points.append(scaled_point.copy())
                    radii.append(radius)

        return np.array(test_points), np.array(radii)

    def evaluate_all(self, posterior_dict, test_points, n_samples=1000):
        """Evaluate all posteriors including reference"""
        observations = []
        results = {'test_points': test_points}

        # Generate observations and reference posteriors
        for theta in tqdm(test_points):
            x_obs = self.simulator(torch.tensor(theta, dtype=torch.float32))
            observations.append(x_obs)

        results['observations'] = observations

        # Reference posteriors
        ref_samples = []
        for x_obs in tqdm(observations, desc="Reference"):
            ref_post = self.task._get_reference_posterior(observation=x_obs.unsqueeze(0))
            ref_samples.append(ref_post.sample((n_samples,)).cpu().numpy())
        results['Reference'] = ref_samples

        # Learned posteriors
        for name, posterior in posterior_dict.items():
            samples = []
            for x_obs in tqdm(observations, desc=name):
                samples.append(posterior.sample((n_samples,), x_obs).cpu().numpy())
            results[name] = samples

        return results

    def c2st(self, X1, X2):
        """C2ST score"""
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import hamming_loss

        X = np.vstack([X1, X2])
        y = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        return hamming_loss(y_test, LogisticRegression(max_iter=1000).fit(X_train, y_train).predict(X_test))

    def compute_c2st_by_radius(self, results, radii):
        """Compute C2ST organized by radius"""
        methods = ['Uniform_NPE', 'TailedNormal_NPE', 'Reference']
        unique_radii = np.unique(np.round(radii, 3))
        c2st_data = {}

        for radius in unique_radii:
            indices = np.where(np.abs(radii - radius) < 1e-3)[0]
            c2st_data[f'{radius:.3f}'] = {}

            for i, m1 in enumerate(methods):
                for m2 in methods[i+1:]:
                    c2st_vals = [self.c2st(results[m1][idx], results[m2][idx]) for idx in indices]
                    c2st_data[f'{radius:.3f}'][f"{m1}_vs_{m2}"] = c2st_vals

        return c2st_data

    def plot_c2st_by_radius(self, c2st_data, dim):
        """Plot average C2ST scores by radius"""
        comparisons = list(next(iter(c2st_data.values())).keys())
        radii = [float(r) for r in c2st_data.keys()]

        plt.figure(figsize=(10, 6))

        for comp in comparisons:
            avg_c2st = []
            for radius_str in c2st_data.keys():
                vals = c2st_data[radius_str][comp]
                avg_c2st.append(np.mean(vals))

            plt.plot(radii, avg_c2st, 'o-', label=comp)

        plt.axhline(0.5, color='gray', linestyle='--', linewidth=2, label='Ideal C2ST=0.5')
        plt.xlabel('Radius')
        plt.ylabel('Average Error Metric [C2ST]')
        plt.title(f'C2ST Performance by Distance from Center ({dim}D)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        os.makedirs('figures', exist_ok=True)
        filename = f'figures/c2st_plot_{dim}d.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        
        plt.show()

def sample_uniform_lhs(n_samples, dim, low=-1.0, high=1.0, device='cpu'):
    """Generate uniform samples using Latin Hypercube Sampling"""
    sampler = qmc.LatinHypercube(d=dim, seed=13)
    u_samples = sampler.random(n_samples)
    # Transform from [0,1]^d to [low,high]^d
    samples = low + (high - low) * u_samples
    return torch.tensor(samples, dtype=torch.float32, device=device)

# Main experiment loop

for dim in dimensions:
    print(f"\n{'='*50}")
    print(f"Running experiment for {dim}D")
    print(f"{'='*50}")

    # Setup
    param_ranges = [(-1.0, 1.0)] * dim
    param_width = 2.0
    sigma_scale = 0.1
    sigmas = [sigma_scale * param_width] * dim
    n_simulations = 4000 * (dim // 2)
    
    # Task
    task = GaussianLinear(dim=dim, prior_scale=param_width/2)
    simulator = task.get_simulator()
    
    # Prior
    prior = ili.utils.IndependentNormal(
        loc=[0.0] * dim,
        scale=sigmas, 
        device=device
    )
    
    # Proposals
    proposal_old = ili.utils.Uniform(
        low=[-1.0] * dim,
        high=[1.0] * dim,
        device=device
    )
    
    proposal_new = TailedNormal(
        a=torch.tensor([-1.0] * dim, dtype=torch.float32),
        b=torch.tensor([1.0] * dim, dtype=torch.float32),
        sigma=torch.tensor(sigmas, dtype=torch.float32)
    )
    
    # Generate training data
    theta_old = sample_uniform_lhs(n_simulations, dim = dim, low=-1.0, high=1.0, device=device)
    theta_new = proposal_new.sample_lhs(n_simulations)
    x_old = simulator(theta_old)
    x_new = simulator(theta_new)
    
    # Create dataloaders
    loader_old = NumpyLoader(x=x_old, theta=theta_old)
    loader_new = NumpyLoader(x=x_new, theta=theta_new)
    
    # Neural networks
    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='maf', hidden_features=16, num_transforms=5,),
        ili.utils.load_nde_sbi(engine='NPE', model='made', hidden_features=16, num_transforms=5,)
    ]
    
    train_args = {
        'training_batch_size': 64,
        'learning_rate': 5e-5
    }
    os.makedirs('toy-n-dim-models', exist_ok=True)
    # Train models
    print("Training Uniform NPE...")
    runner_old = InferenceRunner.load(
        backend='sbi', engine='NPE', prior=prior, nets=nets,
        device=device, train_args=train_args, proposal=proposal_old,
        out_dir=f'toy-n-dim-models/uniform_{dim}d'
    )
    posterior_old, _ = runner_old(loader=loader_old)
    
    print("Training TailedNormal NPE...")
    runner_new = InferenceRunner.load(
        backend='sbi', engine='NPE', prior=prior, nets=nets,
        device=device, train_args=train_args, proposal=proposal_new,
        out_dir=f'toy-n-dim-models/tailed_{dim}d'
    )
    posterior_new, _ = runner_new(loader=loader_new)
    
    # Evaluate
    posterior_dict = {
        'Uniform_NPE': posterior_old,
        'TailedNormal_NPE': posterior_new
    }
    
    print(f"({dim}D)")

    evaluator = CircleEvaluator(simulator, param_ranges, task)
    test_points, radii = evaluator.create_test_points()
    results = evaluator.evaluate_all(posterior_dict, test_points)
    c2st_data = evaluator.compute_c2st_by_radius(results, radii)

    # Plot
    c2st_plot = evaluator.plot_c2st_by_radius(c2st_data, dim)

print("All experiments completed!")
