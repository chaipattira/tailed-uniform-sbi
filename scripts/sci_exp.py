import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import os
from matplotlib.cm import get_cmap
from scipy.stats import truncnorm, qmc
from tqdm import tqdm
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
import emcee

import symbolic_pofk.syren_new as syren_new
import symbolic_pofk.linear as linear
import pickle

# Device configuration
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Device:', device)
sns.set(style="whitegrid", context="paper")

with open('/home/x-ctirapongpra/scratch/sci-2-dim-models/uniform_power/posterior.pkl', 'rb') as f:
    posterior_ensemble_old = pickle.load(f)
with open('/home/x-ctirapongpra/scratch/sci-2-dim-models/tailed_power/posterior.pkl', 'rb') as f:
    posterior_ensemble = pickle.load(f)
with open('/home/x-ctirapongpra/scratch/sci-2-dim-models/nle_power/posterior.pkl', 'rb') as f:
    posterior_ensemble_nle = pickle.load(f)

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

IndependentTailedNormal = type('IndependentTailedNormal', (CustomIndependent,), {'Distribution': TailedNormal})

# Fixed
L, N = 1000, 128
kf = 2*np.pi/L
knyq = np.pi*N/L
kedges = np.arange(0, knyq, kf)
kcenters = (kedges[:-1] + kedges[1:])/2
a = 1.0

def simulator(theta):
   """Simulator: theta -> P(k)"""
   Om, h = theta
   # Also fixed (for now)
   As = 2.105  # 10^9 A_s
   Ob = 0.02242 / h ** 2
   ns = 0.9665
   w0 = -1.0  
   wa = 0.0
   mnu = 0.0
   pk_syren_theory = syren_new.pnl_new_emulated(
         kcenters, As, Om, Ob, h, ns, mnu, w0, wa, a=a
      )

   var_single = np.abs(pk_syren_theory)**2
   Nk = L**3 * kcenters**2 * kf / (2*np.pi**2)
   var_mode = var_single * 2 / Nk
   std_mode = np.sqrt(var_mode)

   pk_w_noise = pk_syren_theory + std_mode*np.random.randn(*pk_syren_theory.shape)
   return pk_w_noise

n_posterior_samples = 10000
n_simulations = 10000

# Parameter ranges [Om, h]
param_1_range = (0.24, 0.40)   # Om
param_2_range = (0.61, 0.73)    # h
param_ranges = [param_1_range, param_2_range]

param_1_width = param_1_range[1] - param_1_range[0]
param_2_width = param_2_range[1] - param_2_range[0]

# Scale sigma relative to parameter ranges
sigma_scale = 0.1
sigmas = [sigma_scale * (high - low) for low, high in param_ranges]

param_1_mean = (param_1_range[0] + param_1_range[1]) / 2  # Om mean
param_2_mean = (param_2_range[0] + param_2_range[1]) / 2  # h mean

param_1_std = 0.1 * (param_1_range[1] - param_1_range[0])   # Om std
param_2_std = 0.1 * (param_2_range[1] - param_2_range[0])   # h std

class CircleEvaluator:
    def __init__(self, simulator, param_ranges):
        self.simulator = simulator
        self.param_ranges = param_ranges
        self.prior_center = np.array([(low + high) / 2 for low, high in param_ranges])
        self.max_radius = min([(high - low) / 2 for low, high in param_ranges])

    def create_test_points(self, n_radii=20, n_angles=30):
        """Create test points on concentric circles"""
        test_points = [self.prior_center.copy()]
        radii = [0.0]

        for radius in np.linspace(0.2 * self.max_radius, 0.9 * self.max_radius, n_radii):
            for angle in np.linspace(0, 2*np.pi, n_angles, endpoint=False):
                x = self.prior_center[0] + radius * np.cos(angle)
                y = self.prior_center[1] + radius * np.sin(angle)

                if (self.param_ranges[0][0] <= x <= self.param_ranges[0][1] and 
                    self.param_ranges[1][0] <= y <= self.param_ranges[1][1]):
                    test_points.append([x, y])
                    radii.append(radius)

        return np.array(test_points), np.array(radii)

    def evaluate_all(self, posterior_dict, posterior_ensemble_nle, test_points, n_samples=1000):
        """Evaluate all posteriors including NLE ensemble"""
        observations = []
        results = {'test_points': test_points}

        # Generate observations for all test points
        for theta in tqdm(test_points, desc="Generating observations"):
            x_obs = self.simulator(theta)
            observations.append(x_obs)

        results['observations'] = observations

        # Sample from NLE ensemble (replaces reference posterior)
        nle_samples = []
        for x_obs in tqdm(observations, desc="NLE Ensemble"):
            samples_nle = posterior_ensemble_nle.sample(
                (n_samples,), x_obs, 
                method='slice_np_vectorized', num_chains=20
            ).detach().cpu().numpy()
            nle_samples.append(samples_nle)
        results['NLE_Ensemble'] = nle_samples

        # Sample from NPE posteriors
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
        methods = ['Uniform_NPE', 'TailedNormal_NPE', 'NLE_Ensemble']
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

    def plot_c2st_by_radius(self, c2st_data):
        """Plot average C2ST scores by radius"""
        comparisons = list(next(iter(c2st_data.values())).keys())
        radii = [float(r) for r in c2st_data.keys()]

        plt.figure(figsize=(10, 6))

        for comp in comparisons:
            avg_c2st = []
            for radius_str in c2st_data.keys():
                vals = c2st_data[radius_str][comp]
                avg_c2st.append(np.mean(vals))

            plt.plot(radii, avg_c2st, 'o-', label=comp.replace('_', ' '))

        plt.axhline(0.5, color='gray', linestyle='--', linewidth=2, label='Ideal C2ST=0.5')
        plt.xlabel('Radius', fontsize=14)
        plt.ylabel('Average Error Metric [C2ST]', fontsize=14)
        plt.title('C2ST Performance by Distance from Center', fontsize=16)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        filename = 'sci-2-dim-figures/c2st-radius-nle.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        plt.show()

    def plot_c2st_tables(self, c2st_data):
        """Plot C2ST as colored tables"""
        sns.set(style="white")
        comparisons = list(next(iter(c2st_data.values())).keys())
        fig, axes = plt.subplots(len(comparisons), 1, figsize=(20, 10*len(comparisons)))
        if len(comparisons) == 1: 
            axes = [axes]

        for i, comp in enumerate(comparisons):
            radii = list(c2st_data.keys())
            max_pts = max(len(c2st_data[r][comp]) for r in radii)

            # Create data matrix
            data = np.full((len(radii), max_pts), np.nan)
            for j, r in enumerate(radii):
                vals = c2st_data[r][comp]
                data[j, :len(vals)] = vals

            # Plot
            im = axes[i].imshow(data, cmap='RdYlBu_r', vmin=0.3, vmax=0.6)

            # Add text
            for j in range(len(radii)):
                for k in range(max_pts):
                    if not np.isnan(data[j, k]):
                        color = 'white' if data[j, k] > 0.5 else 'black'
                        axes[i].text(k, j, f'{data[j, k]:.2f}', ha='center', va='center', 
                                     color=color, fontweight='bold', fontsize=8)

            axes[i].set_yticks(range(len(radii)))
            axes[i].set_yticklabels([f'r={r}' for r in radii], fontsize=10)
            axes[i].set_xlabel('Point Index', fontsize=12)
            axes[i].set_ylabel('Radius', fontsize=12)
            axes[i].set_title(comp.replace('_', ' '), fontsize=14)
            plt.colorbar(im, ax=axes[i], label='C2ST Score')

        plt.tight_layout()
        filename = 'sci-2-dim-figures/c2st-tables-nle.png'
        plt.savefig(filename, bbox_inches='tight')
        print(f"Saved plot: {filename}")
        plt.show()

circle_evaluator = CircleEvaluator(simulator, param_ranges)
test_points, radii = circle_evaluator.create_test_points(n_radii=20, n_angles=30)

# Define posteriors to compare
posterior_dict = {
    'Uniform_NPE': posterior_ensemble_old,
    'TailedNormal_NPE': posterior_ensemble
}

results = circle_evaluator.evaluate_all(posterior_dict, posterior_ensemble_nle, test_points, n_samples=1000)
c2st_data = circle_evaluator.compute_c2st_by_radius(results, radii)
circle_evaluator.plot_c2st_by_radius(c2st_data)
circle_evaluator.plot_c2st_tables(c2st_data)