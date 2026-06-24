#!/bin/bash
#SBATCH --job-name=ext-uniform-dims
#SBATCH --array=0-19
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=2:00:00
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
