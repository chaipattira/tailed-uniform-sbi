#!/bin/bash
#SBATCH --job-name=optuna  # Job name
#SBATCH --array=0-8  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=6:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name

module load anaconda
conda activate tailed-uniform

net_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /home/x-ctirapongpra/scratch/tailed-uniform-sbi

echo "Running optuna inference on job $net_index"

python /home/x-ctirapongpra/scratch/tailed-uniform-sbi/scripts/run_optuna.py
