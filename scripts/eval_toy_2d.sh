#!/bin/bash
#SBATCH --job-name=eval-tails-2d
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --time=0:15:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-tails-2d-%j.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-tails-2d-%j.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "eval_toy_2d starting on $(hostname) at $(date)"
python scripts/eval_toy_2d.py "$@"
echo "eval_toy_2d done at $(date)"
