# Project Context

## What this is

A research project testing whether **tailed proposal priors** improve Neural Posterior Estimator (NPE) quality near parameter-space boundaries in Simulation-Based Inference (SBI).

The central hypothesis: a uniform proposal prior never places simulation budget beyond the prior box, so NPE posteriors are poor at boundary-adjacent and out-of-box parameter values. A tailed proposal prior — which smoothly extends beyond the box — gives the NPE training signal in the tail region and improves coverage there.

See `UBIQUITOUS_LANGUAGE.md` for precise definitions of all domain terms.

## Key distinctions

- **Proposal prior** ≠ **Assumed prior**. The proposal prior is a training-efficiency choice (where to run the simulator). The assumed prior is statistical knowledge (what the NPE uses to form the posterior). They are set independently; changing the proposal does not change the target posterior.
- **GaussianLinear** task uses a **Gaussian assumed prior**. **GaussianLinearUniform** task uses a **Uniform assumed prior**. Both are run with all five proposal variants to disentangle the effect.

## Experiments

| Experiment | Sweep variable | Task | Script pair |
| --- | --- | --- | --- |
| Exp 1 — dims | Dimensionality d ∈ {2,4,8,12,...} | GaussianLinear / GaussianLinearUniform | train_toy_dims / eval_toy_dims |
| Exp 2 — nsims | Number of simulations | GaussianLinear | train_toy_nsims / eval_toy_nsims |
| Exp 3 — sigma | Tail mass σ | GaussianLinear | train_toy_sigma / eval_toy_sigma |
| Cosmology | Real P(k) simulator | (Ω_m, h) uniform | train_cosmo / sample_cosmo_mcmc / eval_cosmo |

## Proposal variants

Five proposal priors are compared in every experiment (all with equal tail mass for fair comparison):

1. `uniform` — flat box, no tails (baseline)
2. `gaussiantailed` — half-normal tails
3. `lineartailed` — linearly-ramping tails
4. `exponentialtailed` — exponentially-decaying tails
5. `uniformtailed` — flat extension of the box

## Code layout

```
toolbox/
  distributions.py   — tailed prior classes (GaussianTailed, LinearTailed, ExponentialTailed, UniformTailed) + Independent wrappers
  priors.py          — cosmology (Ω_m, h) prior factory; also rotated (φ₁, φ₂) priors
  simulators.py      — syren_simulator (P(k) for cosmology); sample_uniform_lhs
  evaluators.py      — C2ST evaluation classes: DistanceEvaluator (primary), RectGridEvaluator, CircleEvaluator, GridEvaluator, SBIEvaluator
  transforms.py      — DegenRotation: log-space (Ω_m, h) ↔ (φ₁, φ₂) rotation
  utils.py           — CPU_Unpickler, load_posterior
  imports.py         — glob import block (used via `from toolbox.imports import *`)
scripts/
  Toy experiments (domain = toy):
    train_toy_{dims,nsims,sigma}.py/.sh — SLURM array; one job per (proposal × sweep_cell)
    eval_toy_{dims,nsims,sigma,2d}.py   — load posteriors, compute C2ST at distance bins, plot
    train_toy_optuna.py                 — Optuna hyperparameter search for toy NPE
  Cosmology experiment (domain = cosmo):
    train_cosmo.py/.sh     — train NPE; --optuna flag runs Optuna search instead of fixed arch
    ensemble_cosmo_optuna.py/.sh — post-Optuna: ensemble top-N trials into one posterior
    sample_cosmo_mcmc.py/.sh    — SLURM array; one job per grid point; MCMC only (no NPE)
    eval_cosmo.py          — single-node; loads MCMC cache, samples NPE, computes C2ST, plots
    submit_cosmo.sh        — orchestrator: chains train → ensemble → sample_mcmc with afterok deps
notebooks-clean/     — analysis and figure notebooks
results/             — trained posterior .pkl files, organized by experiment/dim/proposal
```

## Cosmology pipeline order

```
1. train_cosmo [--optuna]          (SLURM array, one job per proposal)
2. ensemble_cosmo_optuna           (optuna path only; run after inspecting trial results)
3. sample_cosmo_mcmc               (SLURM array, one job per grid point; MCMC only)
4. eval_cosmo                      (single node; NPE sampling + C2ST + plots)
```

`submit_cosmo.sh` automates steps 1–3 with SLURM `afterok` dependencies.

## Evaluation protocol

All experiments use **C2ST vs reference posterior** at **distance bins** from the prior center (center, r=0.25, 0.5, 0.75, 1.0, 2σ-extrapolation). The 2σ-extrapolation bin is the key probe: it places test points outside the prior box, where tailed priors are expected to help. The reference posterior is the analytical (GaussianLinear) or MCMC (cosmology) posterior.

## Known architectural issues

- C2ST logic (`X = vstack([X1,X2]); LogisticRegression.fit(...)`) is copy-pasted across 5 evaluator class methods and 1 standalone function. Two variants exist with different metrics (AUC-ROC vs Hamming loss) — see `UBIQUITOUS_LANGUAGE.md` flagged ambiguities.
- `sample_lhs` is duplicated in every distribution class body.
- `imports.py` acts as a glob import; callers use `from toolbox.imports import *`, obscuring provenance of names like `GaussianLinear`, `NumpyLoader`.
- Plotting logic is embedded in evaluator class methods (tight coupling between computation and visualization).
