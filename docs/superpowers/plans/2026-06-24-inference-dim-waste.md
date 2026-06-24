# inference-dim-waste Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement training scripts, eval script, and analysis notebook to show that ExtendedUniform's posterior quality at the prior boundary degrades faster than TailedUniform's as dimensionality increases under a fixed simulation budget.

**Architecture:** Three-stage pipeline — SLURM array trains 12 models (4 proposals × 3 dims), a sequential eval script loads all posteriors and computes C2ST vs reference at distance bins, a notebook produces the two figures. Mirrors the existing `train_extended_uniform` / `eval_extended_uniform` pattern exactly.

**Tech Stack:** PyTorch, ili (InferenceRunner, NumpyLoader, GaussianLinear), sbi (MAF/MADE via `load_nde_sbi`), scikit-learn (LogisticRegression C2ST), matplotlib/seaborn, SLURM array jobs, Jupyter.

---

## File Map

| File | Action | Responsibility |
|------|--------|----------------|
| `scripts/train_extended_uniform_dims.py` | Create | 12-task SLURM array: task_id → (dim, proposal), trains + saves posterior |
| `scripts/train_extended_uniform_dims.sh` | Create | SLURM array 0-11, mirrors `train_extended_uniform.sh` |
| `scripts/eval_extended_uniform_dims.py` | Create | Loads all posteriors, runs DistanceEvaluator per dim, computes C2ST, saves pkl |
| `scripts/eval_extended_uniform_dims.sh` | Create | Single SLURM job, mirrors `eval_extended_uniform.sh` |
| `notebooks-clean/inference-dim-waste.ipynb` | Create | Loads pkl, produces Figure 1 (per-dim profiles) and Figure 2 (degradation curve) |

Reference files (read, do not modify):
- `scripts/train_extended_uniform.py` — training loop to mirror
- `scripts/eval_extended_uniform.py` — eval loop to mirror
- `toolbox/evaluators.py:607` — `DistanceEvaluator` (works for arbitrary dim as-is)
- `toolbox/distributions.py` — `TailedUniform`
- `toolbox/imports.py` — shared imports (`InferenceRunner`, `NumpyLoader`, `GaussianLinear`, etc.)

---

## Task 1: Training Script

**Files:**
- Create: `scripts/train_extended_uniform_dims.py`

### Task ID Mapping

```
DIMS      = [4, 8, 16]       # dim_idx = 0, 1, 2
PROPOSALS = [                 # proposal_idx = 0, 1, 2, 3
    ('uniform',  None),
    ('tailed',   None),
    ('extended', 0.1),
    ('extended', 0.3),
]
task_id = dim_idx * 4 + proposal_idx   → 12 tasks total (0–11)
```

- [ ] **Step 1: Verify the mapping mentally before writing any code**

Print the full mapping to catch off-by-one errors:
```
task 0  → dim=4,  uniform
task 1  → dim=4,  tailed
task 2  → dim=4,  extended δ=0.1
task 3  → dim=4,  extended δ=0.3
task 4  → dim=8,  uniform
task 5  → dim=8,  tailed
task 6  → dim=8,  extended δ=0.1
task 7  → dim=8,  extended δ=0.3
task 8  → dim=16, uniform
task 9  → dim=16, tailed
task 10 → dim=16, extended δ=0.1
task 11 → dim=16, extended δ=0.3
```

- [ ] **Step 2: Create `scripts/train_extended_uniform_dims.py`**

```python
"""
Train one NPE model for the inference-dim-waste experiment.
Array index → (dim, proposal):
  dim_idx      = task_id // 4   (0→4, 1→8, 2→16)
  proposal_idx = task_id  % 4   (0→uniform, 1→tailed, 2→extended 0.1, 3→extended 0.3)
"""
import argparse
import os
import sys
import random

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.imports import *
from toolbox.distributions import TailedUniform
from toolbox.simulators import sample_uniform_lhs

DIMS = [4, 8, 16]
PROPOSALS = [
    ('uniform',   None),
    ('tailed',    None),
    ('extended',  0.1),
    ('extended',  0.3),
]


def label_for(kind, delta_scale):
    if kind == 'uniform':
        return 'uniform'
    if kind == 'tailed':
        return 'taileduniform'
    return f'extended-uniform-delta-{delta_scale}'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_id', type=int, required=True,
                        help='SLURM_ARRAY_TASK_ID (0-11)')
    parser.add_argument('--n_sims', type=int, default=6000)
    parser.add_argument('--out_root', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'inference-dim-models'))
    args = parser.parse_args()

    dim_idx      = args.task_id // 4
    proposal_idx = args.task_id % 4
    dim          = DIMS[dim_idx]
    kind, delta_scale = PROPOSALS[proposal_idx]
    run_label    = label_for(kind, delta_scale)
    out_dir      = os.path.join(args.out_root, f'dim{dim}', run_label)
    os.makedirs(out_dir, exist_ok=True)

    print(f'[task {args.task_id}] dim={dim}  proposal={run_label}  out={out_dir}')

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    param_ranges = [(-1.0, 1.0)] * dim
    range_width  = 2.0
    sigma        = 0.1 * range_width  # 0.2, per-dimension for TailedUniform

    task      = GaussianLinear(dim=dim, prior_scale=range_width / 2)
    simulator = task.get_simulator()

    prior_narrow = ili.utils.Uniform(
        low =[-1.0] * dim,
        high=[ 1.0] * dim,
        device=device,
    )

    if kind == 'uniform':
        proposal = ili.utils.Uniform(
            low =[-1.0] * dim,
            high=[ 1.0] * dim,
            device=device,
        )
        theta = sample_uniform_lhs(args.n_sims, param_ranges)

    elif kind == 'tailed':
        proposal = TailedUniform(
            a     = torch.tensor([-1.0] * dim, dtype=torch.float32),
            b     = torch.tensor([ 1.0] * dim, dtype=torch.float32),
            sigma = torch.tensor([sigma]  * dim, dtype=torch.float32),
        )
        theta = proposal.sample_lhs(args.n_sims)

    else:  # extended uniform
        delta      = delta_scale * range_width
        ext_low    = -1.0 - delta
        ext_high   =  1.0 + delta
        ext_ranges = [(ext_low, ext_high)] * dim
        proposal   = ili.utils.Uniform(
            low =[ext_low]  * dim,
            high=[ext_high] * dim,
            device=device,
        )
        theta = sample_uniform_lhs(args.n_sims, ext_ranges)

    print(f'θ shape={theta.shape}  range=[{theta.min():.3f}, {theta.max():.3f}]')

    x = simulator(theta)
    print(f'x shape={x.shape}')

    loader = NumpyLoader(x=x, theta=theta)

    nets = [
        ili.utils.load_nde_sbi(engine='NPE', model='maf',  hidden_features=50, num_transforms=5),
        ili.utils.load_nde_sbi(engine='NPE', model='made', hidden_features=50, num_transforms=5),
    ]
    train_args = {'training_batch_size': 64, 'learning_rate': 5e-5}

    runner = InferenceRunner.load(
        backend='sbi', engine='NPE',
        prior=prior_narrow, nets=nets, device=device,
        train_args=train_args, proposal=proposal,
        out_dir=out_dir,
    )
    posterior, summaries = runner(loader=loader)

    import json
    summary_path = os.path.join(out_dir, 'training_summaries.json')
    with open(summary_path, 'w') as f:
        json.dump([{k: list(v) for k, v in s.items()} for s in summaries], f)

    print(f'Done. Posterior saved to {out_dir}/posterior.pkl')


if __name__ == '__main__':
    main()
```

- [ ] **Step 3: Smoke test the mapping logic (no training, no GPU needed)**

```bash
cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi
python -c "
DIMS = [4, 8, 16]
PROPOSALS = [('uniform', None), ('tailed', None), ('extended', 0.1), ('extended', 0.3)]
for tid in range(12):
    d = DIMS[tid // 4]
    k, delta = PROPOSALS[tid % 4]
    label = f'extended-uniform-delta-{delta}' if k == 'extended' else ('taileduniform' if k == 'tailed' else 'uniform')
    print(f'task {tid:2d}: dim={d:2d}  {label}')
"
```

Expected output (12 lines matching the mapping table in Step 1).

- [ ] **Step 4: Quick dry run for task_id=0 with tiny n_sims**

```bash
python scripts/train_extended_uniform_dims.py --task_id 0 --n_sims 50
```

Expected: prints `dim=4  proposal=uniform`, creates `notebooks-clean/inference-dim-models/dim4/uniform/posterior.pkl`, exits cleanly.

- [ ] **Step 5: Commit**

```bash
git add scripts/train_extended_uniform_dims.py
git commit -m "Add train_extended_uniform_dims.py: 12-task SLURM array over dims x proposals"
```

---

## Task 2: SLURM Training Script

**Files:**
- Create: `scripts/train_extended_uniform_dims.sh`

- [ ] **Step 1: Create `scripts/train_extended_uniform_dims.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=ext-uniform-dims
#SBATCH --array=0-11
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=1:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/ext-uniform-dims-%A_%a.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/ext-uniform-dims-%A_%a.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "Array task $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date)"

python scripts/train_extended_uniform_dims.py --task_id $SLURM_ARRAY_TASK_ID

echo "Array task $SLURM_ARRAY_TASK_ID done at $(date)"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/train_extended_uniform_dims.sh
git commit -m "Add train_extended_uniform_dims.sh: SLURM array 0-11"
```

---

## Task 3: Eval Script

**Files:**
- Create: `scripts/eval_extended_uniform_dims.py`

This script loops over all three dims internally (no `--dim` argument). For each dim it instantiates `DistanceEvaluator` (which already handles arbitrary dim via d-dimensional sphere sampling — no subclassing needed), loads posteriors, runs C2ST vs reference, and accumulates results into a single pkl with structure:

```python
{
    4:  {'c2st_results': {...}, 'model_names': [...], 'distance_bin_order': [...]},
    8:  {...},
    16: {...},
}
```

- [ ] **Step 1: Create `scripts/eval_extended_uniform_dims.py`**

```python
"""
Evaluate posteriors for inference-dim-waste experiment.
Loops over dims 4, 8, 16; evaluates C2ST vs reference at distance bins.
Missing posteriors are skipped with a warning.
"""
import argparse
import os
import sys
import pickle
import random

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import hamming_loss
from sklearn.model_selection import train_test_split

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from toolbox.imports import *
from toolbox.evaluators import DistanceEvaluator

DIMS          = [4, 8, 16]
DELTA_SCALES  = [0.1, 0.3]
DISTANCE_BIN_ORDER = ['center', 'r=0.25', 'r=0.5', 'r=0.75', 'r=1.0', '2sigma-extrap']


def c2st(X1, X2):
    X  = np.vstack([X1, X2])
    y  = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))])
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=42)
    return hamming_loss(y_te, LogisticRegression(max_iter=1000).fit(X_tr, y_tr).predict(X_te))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--models_root', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'inference-dim-models'))
    parser.add_argument('--out_dir', type=str,
                        default=os.path.join(ROOT, 'notebooks-clean', 'inference-dim-figures'))
    parser.add_argument('--n_posterior_samples', type=int, default=2000)
    parser.add_argument('--n_points_per_radius',  type=int, default=40)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')
    sns.set(style='whitegrid', context='paper', font_scale=1.2)

    all_results = {}

    for dim in DIMS:
        print(f'\n{"="*60}')
        print(f'dim={dim}')
        print(f'{"="*60}')

        param_ranges = [(-1.0, 1.0)] * dim
        task         = GaussianLinear(dim=dim, prior_scale=1.0)
        simulator    = task.get_simulator()
        evaluator    = DistanceEvaluator(simulator, param_ranges, task)
        print(f'σ={evaluator.sigma:.4f}  2σ={2*evaluator.sigma:.4f}')

        # Build model path dict
        model_paths = {
            'Uniform':       os.path.join(args.models_root, f'dim{dim}', 'uniform',       'posterior.pkl'),
            'TailedUniform': os.path.join(args.models_root, f'dim{dim}', 'taileduniform', 'posterior.pkl'),
        }
        for d in DELTA_SCALES:
            model_paths[f'ExtUniform δ={d}'] = os.path.join(
                args.models_root, f'dim{dim}', f'extended-uniform-delta-{d}', 'posterior.pkl')

        posteriors = {}
        for name, path in model_paths.items():
            try:
                with open(path, 'rb') as f:
                    posteriors[name] = pickle.load(f)
                print(f'Loaded: {name}')
            except FileNotFoundError:
                print(f'NOT FOUND (skipped): {path}')

        if not posteriors:
            print(f'No posteriors found for dim={dim}, skipping.')
            continue

        test_points, distance_bins = evaluator.create_test_points(
            n_points_per_radius=args.n_points_per_radius)
        print(f'Test points: {len(test_points)}')

        results_all = evaluator.evaluate_all(
            posteriors, test_points, n_samples=args.n_posterior_samples)

        model_names = [k for k in results_all if k not in ('test_points', 'observations', 'Reference')]
        c2st_results = {}
        for name in model_names:
            c2st_results[name] = {}
            for label in DISTANCE_BIN_ORDER:
                idx  = np.where(distance_bins == label)[0]
                vals = [c2st(results_all[name][i], results_all['Reference'][i]) for i in idx]
                c2st_results[name][label] = vals

        all_results[dim] = {
            'c2st_results':      c2st_results,
            'model_names':       model_names,
            'distance_bin_order': DISTANCE_BIN_ORDER,
        }

        # Per-dim summary
        print(f'\nC2ST vs Reference — dim={dim}')
        print(f'{"":22s}', end='')
        for label in DISTANCE_BIN_ORDER:
            print(f'{label:20s}', end='')
        print()
        for name in model_names:
            print(f'{name:22s}', end='')
            for label in DISTANCE_BIN_ORDER:
                vals = np.array(c2st_results[name][label])
                if len(vals):
                    print(f'{np.mean(vals):.3f}±{np.std(vals):.3f}      ', end='')
                else:
                    print(f'{"N/A":20s}', end='')
            print()

    # Save combined results
    out_path = os.path.join(args.out_dir, 'inference-dims-c2st-results.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(all_results, f)
    print(f'\nSaved combined results to {out_path}')


if __name__ == '__main__':
    main()
```

- [ ] **Step 2: Smoke test (import check — no models needed)**

```bash
python -c "import scripts.eval_extended_uniform_dims" 2>/dev/null || \
python scripts/eval_extended_uniform_dims.py --help
```

Expected: prints usage/help without error.

- [ ] **Step 3: Commit**

```bash
git add scripts/eval_extended_uniform_dims.py
git commit -m "Add eval_extended_uniform_dims.py: sequential C2ST eval across dims"
```

---

## Task 4: SLURM Eval Script

**Files:**
- Create: `scripts/eval_extended_uniform_dims.sh`

- [ ] **Step 1: Create `scripts/eval_extended_uniform_dims.sh`**

```bash
#!/bin/bash
#SBATCH --job-name=eval-ext-uniform-dims
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=3:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-ext-uniform-dims-%j.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-ext-uniform-dims-%j.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "Job $SLURM_JOB_ID starting on $(hostname) at $(date)"

python scripts/eval_extended_uniform_dims.py "$@"

echo "Job $SLURM_JOB_ID done at $(date)"
```

- [ ] **Step 2: Commit**

```bash
git add scripts/eval_extended_uniform_dims.sh
git commit -m "Add eval_extended_uniform_dims.sh: single SLURM job for sequential eval"
```

---

## Task 5: Analysis Notebook

**Files:**
- Create: `notebooks-clean/inference-dim-waste.ipynb`

The notebook loads `inference-dims-c2st-results.pkl` and produces two saved figures.

- [ ] **Step 1: Create notebook with Cell 1 — imports and config**

```python
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid', context='paper', font_scale=1.2)

ROOT     = os.path.dirname(os.path.abspath(''))  # project root when run from notebooks-clean/
FIGURES  = os.path.join(ROOT, 'notebooks-clean', 'inference-dim-figures')
PKL_PATH = os.path.join(FIGURES, 'inference-dims-c2st-results.pkl')

with open(PKL_PATH, 'rb') as f:
    all_results = pickle.load(f)

DIMS               = sorted(all_results.keys())          # [4, 8, 16]
DISTANCE_BIN_ORDER = all_results[DIMS[0]]['distance_bin_order']

# Theoretical waste fractions: 1 - (1/(1+delta))^d
WASTE = {
    ('extended', 0.1): {d: 1 - (1/1.1)**d for d in DIMS},
    ('extended', 0.3): {d: 1 - (1/1.3)**d for d in DIMS},
}

print('Loaded dims:', DIMS)
print('Bins:', DISTANCE_BIN_ORDER)
```

- [ ] **Step 2: Add Cell 2 — style map**

```python
STYLE_MAP = {
    'Uniform':        {'color': '#555555', 'marker': 'o',  'ls': '-',  'lw': 2.5},
    'TailedUniform':  {'color': '#e07b00', 'marker': 's',  'ls': '--', 'lw': 2.5},
    'ExtUniform δ=0.1': {'color': '#2676ae', 'marker': 'D', 'ls': ':',  'lw': 2.0},
    'ExtUniform δ=0.3': {'color': '#0d3b6e', 'marker': 'D', 'ls': ':',  'lw': 2.0},
}
```

- [ ] **Step 3: Add Cell 3 — Figure 1: per-dim C2ST profiles**

```python
fig, axes = plt.subplots(1, len(DIMS), figsize=(5 * len(DIMS), 5), sharey=True)

x_pos = np.arange(len(DISTANCE_BIN_ORDER))

for col, dim in enumerate(DIMS):
    ax = axes[col]
    res = all_results[dim]
    model_names = res['model_names']
    offsets = np.linspace(-0.3, 0.3, len(model_names))

    for i, name in enumerate(model_names):
        means, stds, xs = [], [], []
        for j, label in enumerate(DISTANCE_BIN_ORDER):
            vals = np.array(res['c2st_results'][name][label])
            if len(vals):
                means.append(np.mean(vals))
                stds.append(np.std(vals))
                xs.append(x_pos[j] + offsets[i])
        s = STYLE_MAP.get(name, {'color': 'gray', 'marker': 'o'})
        ax.errorbar(xs, means, yerr=stds, fmt=s['marker'],
                    color=s['color'], label=name,
                    capsize=3, capthick=1.5, linewidth=0,
                    elinewidth=1.5, markersize=7, alpha=0.9)

    ax.axvline(4.5, color='red', linestyle='--', linewidth=1.5, alpha=0.8,
               label='Prior boundary')
    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1.5, alpha=0.7)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['Ctr', '0.25', '0.5', '0.75', '1.0', '2σ'],
                       fontsize=9, rotation=30)
    ax.set_title(f'dim={dim}', fontsize=13)
    ax.set_xlabel('Distance from prior center', fontsize=11)
    if col == 0:
        ax.set_ylabel('C2ST vs Reference', fontsize=11)
    if col == len(DIMS) - 1:
        ax.legend(fontsize=8, loc='upper left')

fig.suptitle('Proposal efficiency across dimensions (6000 sims fixed)', fontsize=14)
plt.tight_layout()
os.makedirs(FIGURES, exist_ok=True)
fig.savefig(os.path.join(FIGURES, 'dim-waste-c2st-profiles.pdf'), bbox_inches='tight', dpi=300)
plt.show()
print('Saved dim-waste-c2st-profiles.pdf')
```

- [ ] **Step 4: Add Cell 4 — Figure 2: degradation curve at 2σ-extrap**

```python
fig, ax = plt.subplots(figsize=(7, 5))

# Collect C2ST at extrapolation bin per (proposal, dim)
EXTRAP_BIN = '2sigma-extrap'

for name, s in STYLE_MAP.items():
    means, stds = [], []
    dims_available = []
    for dim in DIMS:
        if dim not in all_results:
            continue
        vals = np.array(all_results[dim]['c2st_results'].get(name, {}).get(EXTRAP_BIN, []))
        if len(vals):
            means.append(np.mean(vals))
            stds.append(np.std(vals))
            dims_available.append(dim)
    if means:
        ax.errorbar(dims_available, means, yerr=stds,
                    fmt=f"{s['marker']}{s['ls']}",
                    color=s['color'], label=name,
                    capsize=4, capthick=1.5, linewidth=s['lw'],
                    elinewidth=1.5, markersize=8, alpha=0.95)

# Annotate waste fractions for ExtUniform
for delta, delta_key in [(0.1, 'ExtUniform δ=0.1'), (0.3, 'ExtUniform δ=0.3')]:
    for dim in DIMS:
        waste = 1 - (1 / (1 + delta)) ** dim
        if dim not in all_results:
            continue
        vals = np.array(all_results[dim]['c2st_results'].get(delta_key, {}).get(EXTRAP_BIN, []))
        if len(vals):
            ax.annotate(f'{waste*100:.0f}%',
                        xy=(dim, np.mean(vals)),
                        xytext=(3, 4), textcoords='offset points',
                        fontsize=7, color=STYLE_MAP[delta_key]['color'], alpha=0.8)

ax.axhline(0.5, color='gray', linestyle=':', linewidth=2, alpha=0.7, label='Ideal C2ST=0.5')
ax.set_xlabel('Dimensionality', fontsize=13)
ax.set_ylabel('C2ST vs Reference (2σ extrapolation)', fontsize=13)
ax.set_title('Degradation at prior boundary: fixed 6000 sims\n(% = wasted simulations outside prior)', fontsize=12)
ax.set_xticks(DIMS)
ax.legend(fontsize=10)
plt.tight_layout()
fig.savefig(os.path.join(FIGURES, 'dim-waste-degradation.pdf'), bbox_inches='tight', dpi=300)
plt.show()
print('Saved dim-waste-degradation.pdf')
```

- [ ] **Step 5: Commit**

```bash
git add notebooks-clean/inference-dim-waste.ipynb
git commit -m "Add inference-dim-waste.ipynb: per-dim C2ST profiles and degradation curve"
```

---

## Running the Full Pipeline

After all tasks are committed:

```bash
# 1. Submit training array (12 jobs)
sbatch scripts/train_extended_uniform_dims.sh

# 2. Once all 12 jobs complete, run eval
sbatch scripts/eval_extended_uniform_dims.sh

# 3. Open notebook and run all cells
jupyter nbconvert --to notebook --execute notebooks-clean/inference-dim-waste.ipynb
```

Check output:
- `notebooks-clean/inference-dim-models/dim{4,8,16}/*/posterior.pkl` — 12 files
- `notebooks-clean/inference-dim-figures/inference-dims-c2st-results.pkl`
- `notebooks-clean/inference-dim-figures/dim-waste-c2st-profiles.pdf`
- `notebooks-clean/inference-dim-figures/dim-waste-degradation.pdf`
