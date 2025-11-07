#!/bin/bash
#SBATCH --job-name=optuna  # Job name
#SBATCH --array=0-8  # Array range
#SBATCH --nodes=1               # Number of nodes
#SBATCH --ntasks=4            # Number of tasks
#SBATCH --time=4:00:00         # Time limit
#SBATCH --partition=shared  # Partition name
#SBATCH --account=phy240043  # Account name

conda activate your-env-name

net_index=$SLURM_ARRAY_TASK_ID

# Command to run for each lhid
cd /path/to/your/tailed-normal-sbi/scripts

echo "Running optuna inference on job $net_index"

python -m run_optuna.py
