#!/bin/bash
#SBATCH --job-name=cosmo-grid
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=0:10:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --array=0-99
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/cosmo-grid-%A_%a.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/cosmo-grid-%A_%a.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "cosmo-grid task ${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"
python scripts/sample_cosmo_mcmc.py --task_id ${SLURM_ARRAY_TASK_ID} --no-rotated \
    --out_root results/science/original/grid_samples_10x10 "$@"
echo "cosmo-grid task ${SLURM_ARRAY_TASK_ID} done at $(date)"
