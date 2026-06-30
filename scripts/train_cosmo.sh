#!/bin/bash
#SBATCH --job-name=train-cosmo
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=12:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --array=0-4
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/train-cosmo-%A_%a.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/train-cosmo-%A_%a.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "train_cosmo task=${SLURM_ARRAY_TASK_ID} starting on $(hostname) at $(date)"
python scripts/train_cosmo.py --task_id ${SLURM_ARRAY_TASK_ID} "$@"
echo "train_cosmo task=${SLURM_ARRAY_TASK_ID} done at $(date)"
