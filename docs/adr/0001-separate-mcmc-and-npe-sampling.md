# ADR 0001 — Separate MCMC and NPE sampling into distinct pipeline stages

**Status:** Accepted

## Context

`sample_cosmo_grid.py` (now `sample_cosmo_mcmc.py`) originally ran both emcee MCMC (reference posterior) and NPE posterior sampling for every grid point in one SLURM array job. When new NPE models arrived (e.g., optuna-tuned ensemble), a bespoke `resample_cosmo_optuna.py` script was needed to re-run NPE sampling against the existing MCMC cache without re-running MCMC.

## Decision

MCMC sampling and NPE sampling are separate pipeline stages:

- `sample_cosmo_mcmc` — MCMC only. Runs emcee per grid point, saves `obs_list` and `mcmc_list`. Stable: never needs to be re-run unless the simulator, prior, or grid changes.
- `eval_cosmo` — NPE sampling + C2ST + plotting. Reads the MCMC cache, samples each NPE model, computes C2ST, and produces figures in one sequential pass on a single node.

## Consequences

- `resample_cosmo_optuna.py` is deleted — resampling NPE with new models is just re-running `eval_cosmo`.
- `eval_cosmo_degen.py` and `plot_cosmo_c2st.py` are merged into `eval_cosmo.py`.
- `eval_cosmo` does not need a SLURM array because NPE sampling is fast (forward pass only).
- MCMC and NPE sample counts must be set consistently in `sample_cosmo_mcmc`; there is no downstream resampling mechanism.
