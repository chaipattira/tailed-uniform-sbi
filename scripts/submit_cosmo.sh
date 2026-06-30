#!/bin/bash
# Submit the cosmology pipeline (optuna path):
#   1. train_cosmo --optuna   (array 0-4)
#   2. ensemble_cosmo_optuna  (array 0-4, afterok: train)
#   3. sample_cosmo_mcmc      (array 0-48, afterok: ensemble, --no-rotated)
#
# Run eval_cosmo manually after step 3 completes (single node, no SLURM needed).
#
# Usage: bash scripts/submit_cosmo.sh [extra args for train/ensemble scripts]
#   e.g. bash scripts/submit_cosmo.sh --n_sims 3000 --n_top 5

set -e

OPTUNA_ROOT="/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/results/science/original/optuna"

TRAIN_JID=$(sbatch --parsable scripts/train_cosmo.sh --optuna "$@")
echo "Submitted train job:    ${TRAIN_JID}"

ENSEMBLE_JID=$(sbatch --parsable \
    --dependency=afterok:${TRAIN_JID} \
    scripts/ensemble_cosmo_optuna.sh "$@")
echo "Submitted ensemble job: ${ENSEMBLE_JID} (depends on ${TRAIN_JID})"

MCMC_JID=$(sbatch --parsable \
    --dependency=afterok:${ENSEMBLE_JID} \
    scripts/sample_cosmo_mcmc.sh --no-rotated --models_root ${OPTUNA_ROOT})
echo "Submitted mcmc job:     ${MCMC_JID} (depends on ${ENSEMBLE_JID})"

echo ""
echo "After mcmc job completes, run eval manually:"
echo "  python scripts/eval_cosmo.py --no-rotated --models_root ${OPTUNA_ROOT}"
