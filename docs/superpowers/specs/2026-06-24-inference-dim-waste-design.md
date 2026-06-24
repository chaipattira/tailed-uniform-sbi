# Design: High-Dimensional Proposal Efficiency Experiment

**Date:** 2026-06-24
**Notebook:** `notebooks-clean/inference-dim-waste.ipynb`

## Goal

Demonstrate that with a fixed simulation budget (6000 sims), ExtendedUniform's posterior
quality at the prior boundary degrades faster than TailedUniform's as dimensionality
increases — because the fraction of simulations wasted outside the region of interest
grows as `1 - (1/(1+δ))^d`.

Theoretical waste fractions:
| dim | δ=0.1 | δ=0.3 |
|-----|-------|-------|
| 4   | 31.7% | 65.0% |
| 8   | 53.3% | 87.7% |
| 16  | 78.2% | 98.5% |

## Files

```
scripts/train_extended_uniform_dims.py   ← SLURM array, 12 tasks
scripts/train_extended_uniform_dims.sh
scripts/eval_extended_uniform_dims.py    ← sequential eval, saves .pkl
scripts/eval_extended_uniform_dims.sh
notebooks-clean/inference-dim-waste.ipynb
notebooks-clean/inference-dim-models/dim{4,8,16}/{label}/posterior.pkl
notebooks-clean/inference-dim-figures/inference-dims-c2st-results.pkl
```

## Training Script (`train_extended_uniform_dims.py`)

### Task ID mapping

12 tasks: `task_id = dim_idx * 4 + proposal_idx`

```
DIMS      = [4, 8, 16]
PROPOSALS = [
    ('uniform',   None),
    ('tailed',    None),    # σ = 0.1 × range_width = 0.2 per dim
    ('extended',  0.1),     # δ = 0.1 × range_width = 0.2 per dim
    ('extended',  0.3),     # δ = 0.3 × range_width = 0.6 per dim
]
```

### Per-task config

- `GaussianLinear(dim=d, prior_scale=1.0)`
- `param_ranges = [(-1.0, 1.0)] * d`
- `prior_narrow = Uniform(low=[-1]*d, high=[1]*d)`
- `n_sims = 6000`, `SEED = 42`
- Architecture: MAF + MADE, `hidden_features=50`, `num_transforms=5` (fixed across all
  dims to isolate proposal effect, not network capacity)
- Output dir: `notebooks-clean/inference-dim-models/dim{d}/{label}/`

### Label scheme

- `uniform` → `uniform`
- `tailed` → `taileduniform`
- `extended, δ` → `extended-uniform-delta-{δ}`

## Eval Script (`eval_extended_uniform_dims.py`)

Runs sequentially (not a SLURM array). For each dim:

1. Instantiate `GaussianLinear(dim=d)` and `DistanceEvaluator(simulator, [(-1,1)]*d, task)`
2. Load all 4 posteriors from `inference-dim-models/dim{d}/`
3. Call `evaluator.create_test_points(n_points_per_radius=40)` — same distance bins as 2D:
   `center, r=0.25, r=0.5, r=0.75, r=1.0, 2σ-extrap` (σ ≈ 0.577, independent of dim)
4. Call `evaluator.evaluate_all(...)` to draw posterior samples and reference samples
5. Compute C2ST vs Reference per (proposal, bin)

Output: one `inference-dims-c2st-results.pkl` with structure:
```python
{
  dim: {
    'c2st_results': {proposal_name: {bin_label: [c2st_vals]}},
    'model_names': [...],
    'distance_bin_order': [...],
  }
}
```

## Notebook (`inference-dim-waste.ipynb`)

Loads `inference-dims-c2st-results.pkl` and produces two figures saved to
`inference-dim-figures/`:

### Figure 1 — Per-dim C2ST profiles (`dim-waste-c2st-profiles.pdf`)
1×3 subplot (columns = dim 4, 8, 16). Each panel: C2ST vs distance bin for all 4
proposals, styled like `extended-uniform-c2st-all.pdf`. Red dashed line at prior
boundary, gray dashed line at C2ST=0.5.

### Figure 2 — Degradation curve (`dim-waste-degradation.pdf`)
C2ST at `2σ-extrap` bin vs dim (x-axis: 4, 8, 16) for each proposal. Secondary
annotation: theoretical waste fraction per (proposal, dim). This is the primary
claim-A figure.

## Connections to Existing Code

- `DistanceEvaluator` in `toolbox/evaluators.py` already handles arbitrary `dim` via
  d-dimensional Normal sphere sampling — no changes needed.
- `GaussianLinear` from `ili` supports arbitrary `dim`.
- Training loop mirrors `train_extended_uniform.py` exactly; only the dim loop and
  output path differ.
- Eval loop mirrors `eval_extended_uniform.py`; `ExtendedDistanceEvaluator` subclass
  is not needed since the base `DistanceEvaluator` bins already include `2σ-extrap`.
