#!/bin/bash
#SBATCH --job-name=eval-cosmo
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=2:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-cosmo-%j.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-cosmo-%j.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "eval_cosmo starting on $(hostname) at $(date)"
python scripts/eval_cosmo.py "$@"
echo "eval_cosmo done at $(date)"
