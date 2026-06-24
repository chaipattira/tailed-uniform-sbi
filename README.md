# Learning at the Edge: Tailed-Uniform Sampling for Robust Simulation-Based Inference

Neural posterior estimators trained with uniform priors fail near parameter space boundaries due to sharp density discontinuities that arise when the proposal distribution abruptly drops to zero. We introduce the **Tailed-Uniform** distribution to address this pathology, enabling reliable posterior estimation across the full parameter domain—including regions near and beyond the prior bounds.

**[Read our PAI26 paper](https://openreview.net/pdf?id=3SpOHc1NNG)**

---

## Overview

Simulation-based inference (SBI) typically draws training parameters from a uniform prior over a bounded region $\Theta$. At the boundary $\partial\Theta$, the distribution exhibits a discontinuous step from a flat density to zero. This discontinuity is pathological for neural posterior estimators, which must interpolate across a training distribution that provides no information about how the density behaves just beyond the bounds. The consequence is systematically degraded posterior quality for parameters near boundary regions—a problem that worsens with the dimensionality of the parameter space, as boundary effects grow to dominate an increasing fraction of the prior volume.

The Tailed-Uniform proposal resolves this by augmenting the uniform core with smooth half-normal tails that extend beyond the parameter bounds. The resulting distribution is continuous everywhere, provides training coverage beyond $\partial\Theta$, and gives the neural network the context needed to produce accurate posteriors throughout the bounded region.

---

## Method

### The Boundary Pathology

Standard SBI samples parameters uniformly within a bounded region:

$$\theta \sim \mathcal{U}([\theta_{\text{min}}, \theta_{\text{max}}])$$

At each boundary, the proposal density drops sharply from a positive constant to zero. Neural posterior estimators trained on such data learn a density ratio that is poorly constrained at $\partial\Theta$, producing unreliable posteriors for any test point whose true posterior has support near or beyond the boundary.

### Tailed-Uniform Distribution

The Tailed-Uniform distribution attaches half-normal tails to the uniform core, yielding a proposal that is continuous and smooth at the boundaries:

$$
\tilde{\mathcal{P}}_{\text{TailedUniform}}(x;\, a, b, \sigma) = \begin{cases}
A \cdot \mathcal{N}(a, \sigma^2), & x \leq a \\
B \cdot \mathcal{U}(a, b), & x \in [a, b] \\
A \cdot \mathcal{N}(b, \sigma^2), & x \geq b
\end{cases}
$$

where the normalization constants ensure continuity at $x = a$ and $x = b$:

$$A = \frac{\sqrt{2\pi\sigma^2}}{\sqrt{2\pi\sigma^2} + (b-a)}, \quad B = \frac{b-a}{\sqrt{2\pi\sigma^2} + (b-a)}$$

For multivariate parameters $\boldsymbol{\theta} \in \mathbb{R}^d$, the distribution is defined via independent marginals:

$$\tilde{\mathcal{P}}_{\text{TailedUniform}}(\boldsymbol{\theta};\, \mathbf{a}, \mathbf{b}, \boldsymbol{\sigma}) = \prod_{i=1}^{d}\tilde{\mathcal{P}}_{\text{TailedUniform}}(\theta_i;\, a_i, b_i, \sigma_i)$$

The tail width $\sigma$ is the single free hyperparameter. In practice, setting $\sigma$ to roughly 10% of the prior range per dimension is robust across the experiments we consider.

We find that replacing the uniform proposal with Tailed-Uniform  consistently improves posterior quality, as measured by the classifier two-sample test (C2ST).
The advantage is concentrated near boundary regions and appears to grow with dimension, as boundary effects scale with the surface-area-to-volume ratio of the bounded parameter space.

---

## Usage

The Tailed-Uniform distribution integrates directly into any SBI workflow that accepts a custom proposal. The example below uses [LtU-ILI](https://github.com/maho3/ltu-ili):

```python
from toolbox.distributions import IndependentTailedUniform
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
import torch

# Define parameter bounds and tail widths
a = torch.tensor([a1, a2])
b = torch.tensor([b1, b2])
sigma = 0.1 * (b - a)  # 10% of the prior range per dimension

# Create the tailed-uniform proposal
proposal = IndependentTailedUniform(a=a, b=b, sigma=sigma)

# Sample parameters and run the forward simulator
theta = proposal.sample((10000,))
x = run_simulator(theta)

# Train a neural posterior estimator with LtU-ILI
loader = NumpyLoader(x=x, theta=theta)
runner = InferenceRunner.load(
    backend='lampe', engine='NPE',
    prior=prior, proposal=proposal,
    nets=nets, out_dir='models/'
)
posterior, _ = runner(loader=loader)
```

---

## Installation

```bash
conda create -n tailed-uniform python=3.10 -y
conda activate tailed-uniform
git clone https://github.com/maho3/ltu-ili.git
cd ltu-ili
pip install ".[pytorch]"
cd ..
pip install sbibm emcee optuna
pip install git+https://github.com/DeaglanBartlett/symbolic_pofk.git
python -m ipykernel install --user --name tailed-uniform --display-name "tailed-uniform"
```

---

## Reproducing the Experiments

### Toy Experiment: 2D Gaussian Linear

The toy experiment trains neural posterior estimators on a two-dimensional Gaussian linear model and evaluates posterior quality via C2ST across a spatial grid of test points.

1. **Train** models with uniform and tailed-uniform proposals: `notebooks-clean/toy-2-dim-training.ipynb`
2. **Evaluate** posteriors at a single test point (corner plots): [toy-2-dim-inference-corner.ipynb](notebooks-clean/toy-2-dim-inference-corner.ipynb)
3. **Evaluate** posteriors across parameter space (spatial heatmaps): [toy-2-dim-inference-spatial.ipynb](notebooks-clean/toy-2-dim-inference-spatial.ipynb)

### Ablation Studies

The following notebooks isolate the effect of each axis of variation:

- [inference-sigmas.ipynb](notebooks-clean/inference-sigmas.ipynb) — sensitivity to tail width $\sigma$
- [inference-nsims.ipynb](notebooks-clean/inference-nsims.ipynb) — performance as a function of simulation budget
- [inference-dimensions.ipynb](notebooks-clean/inference-dimensions.ipynb) — scaling behavior with parameter space dimension

### Science Experiment: Cosmological Parameter Inference

The science experiment applies tailed-uniform sampling to inferring the matter density $\Omega_m$ and dimensionless Hubble constant $h$ from the matter power spectrum $P(k)$.

1. **Hyperparameter optimization** — configure Optuna studies: [sci-2-dim-optuna.ipynb](notebooks-clean/sci-2-dim-optuna.ipynb)
2. **Parallel search** — run distributed optimization on a SLURM cluster:
   ```bash
   sbatch scripts/run_optuna.sh
   ```
3. **Ground truth** — generate MCMC reference posteriors: [sci-2-dim-mcmc.ipynb](notebooks-clean/sci-2-dim-mcmc.ipynb)
4. **Analysis** — compare NPE and MCMC posteriors: [sci-2-dim-inference.ipynb](notebooks-clean/sci-2-dim-inference.ipynb)



