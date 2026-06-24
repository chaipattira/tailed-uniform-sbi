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
