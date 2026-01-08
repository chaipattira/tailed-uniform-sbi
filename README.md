# Learning at the Edge: Tailed-Uniform Sampling for Robust Simulation-Based Inference

Neural posterior estimators trained with uniform sampling fail near parameter space boundaries due to sharp density discontinuities.

- Paper: [tailed_uniform_paper.pdf](paper/tailed_uniform_paper.pdf)
## Main Claim

**Tailed-Uniform** eliminates boundary pathologies in neural posterior estimation by extending uniform distributions with smooth Gaussian tails beyond parameter bounds. This approach:
- Outperforms uniform sampling even with **8× fewer simulations**
- Maintains superior posterior quality near boundaries while preserving performance in interior regions
- Shows increasing advantage in higher dimensions where boundaries dominate parameter space volume

## Method

### Problem
Standard simulation-based inference samples parameters uniformly within bounded regions:

$$\theta \sim \mathcal{U}([\theta_{\text{min}}, \theta_{\text{max}}])$$

This creates sharp discontinuities at boundaries where density collapses to zero, causing neural networks to produce poor posterior estimates near $\partial\Theta$.

### Tailed-Uniform Distribution

The **Tailed-Uniform** distribution extends uniform sampling with half-normal tails:

$$
\tilde{\mathcal{P}}_{\text{TailedUniform}}(x; a, b, \sigma) = \begin{cases}
A \cdot \mathcal{N}(a, \sigma^2), & x \leq a \\
B \cdot \mathcal{U}(a, b), & x \in [a, b] \\
A \cdot \mathcal{N}(b, \sigma^2), & x \geq b
\end{cases}
$$

where normalization constants ensure continuity:

$$A = \frac{\sqrt{2\pi\sigma^2}}{\sqrt{2\pi\sigma^2} + (b-a)}, \quad B = \frac{b-a}{\sqrt{2\pi\sigma^2} + (b-a)}$$

**Multivariate extension:** For $\boldsymbol{\theta} \in \mathbb{R}^d$, use independent marginals:

$$\tilde{\mathcal{P}}_{\text{TailedUniform}}(\boldsymbol{\theta}; \mathbf{a}, \mathbf{b}, \boldsymbol{\sigma}) = \prod_{i=1}^{d}\tilde{\mathcal{P}}_{\text{TailedUniform}}(\theta_i; a_i, b_i, \sigma_i)$$

**Hyperparameter guidance:** Set tail width as 10-40% of parameter range:

$$\sigma_i = \alpha \cdot (b_i - a_i), \quad \alpha \in [0.1, 0.4]$$

### Usage Example

Our **Tailed-Uniform** distribution seamlessly integrates into standard SBI workflows:

```python
from toolbox.distributions import IndependentTailedUniform
from ili.inference import InferenceRunner
from ili.dataloaders import NumpyLoader
import torch

# Define parameter bounds and tail widths
a = torch.tensor([a1, a2])
b = torch.tensor([b1, b2])
sigma = 0.1 * (b - a)  # 10% of range width

# Create tailed-uniform proposal
proposal = IndependentTailedUniform(a=a, b=b, sigma=sigma)

# Sample parameters and run simulator
theta = proposal.sample((10000,))
x = run_simulator(theta)  # Your forward model

# Train neural posterior with LtU-ILI
loader = NumpyLoader(x=x, theta=theta)
runner = InferenceRunner.load(
    backend='lampe', engine='NPE',
    prior=prior, proposal=proposal,
    nets=nets, out_dir='models/'
)
posterior, _ = runner(loader=loader)
```

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

## Getting Started

### Toy Experiment (2D Gaussian Linear)

1. Train models with both uniform and tailed-uniform proposals:
   ```
   notebooks-clean/toy-2-dim-training.ipynb
   ```

2. Analyze posterior quality:
   - [toy-2-dim-inference-corner.ipynb](notebooks-clean/toy-2-dim-inference-corner.ipynb) - Corner plots comparing a single test point
   - [toy-2-dim-inference-spatial.ipynb](notebooks-clean/toy-2-dim-inference-spatial.ipynb) - Spatial analysis acorss space

### Ablation Studies

- [inference-sigmas.ipynb](notebooks-clean/inference-sigmas.ipynb) - Effect of tail width $\sigma$
- [inference-nsims.ipynb](notebooks-clean/inference-nsims.ipynb) - Effect of simulation budget
- [inference-dimensions.ipynb](notebooks-clean/inference-dimensions.ipynb) - Effect of higher dimensions


### Science Experiment (Cosmological Parameter Inference)

Infer matter density $\Omega_m$ and dimensionless Hubble $h$ from matter power spectrum $P(k)$:

1. Set up Optuna studies for hyperparameter optimization:
   ```
   notebooks-clean/sci-2-dim-optuna.ipynb
   ```

2. Run parallel hyperparameter search (requires SLURM cluster):
   ```bash
   sbatch scripts/run_optuna.sh
   ```

3. Analyze models and compare against MCMC ground truth:
   ```
   notebooks-clean/sci-2-dim-inference.ipynb
   ```



---


