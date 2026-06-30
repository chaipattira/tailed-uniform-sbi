#!/bin/bash
#SBATCH --job-name=eval-tails-nsims
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=1:00:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-tails-nsims-%j.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/eval-tails-nsims-%j.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "eval_toy_nsims starting on $(hostname) at $(date)"
python scripts/eval_toy_nsims.py "$@"
echo "eval_toy_nsims done at $(date)"
