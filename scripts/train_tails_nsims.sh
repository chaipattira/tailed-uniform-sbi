#!/bin/bash
#SBATCH --job-name=tails-nsims
#SBATCH --array=0-39
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=0:45:00
#SBATCH --partition=shared
#SBATCH --account=phy240043
#SBATCH --output=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/tails-nsims-%A_%a.out
#SBATCH --error=/anvil/scratch/x-ctirapongpra/tailed-uniform-sbi/jobout/tails-nsims-%A_%a.err

module load anaconda
conda activate tailed-uniform

cd /anvil/scratch/x-ctirapongpra/tailed-uniform-sbi

echo "Array task $SLURM_ARRAY_TASK_ID starting on $(hostname) at $(date)"

python scripts/train_tails_nsims.py --task_id $SLURM_ARRAY_TASK_ID "$@"

echo "Array task $SLURM_ARRAY_TASK_ID done at $(date)"
