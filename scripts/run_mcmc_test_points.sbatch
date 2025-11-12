#!/bin/bash
#SBATCH --job-name=mcmc_test       # Job name
#SBATCH --nodes=1                  # Number of nodes
#SBATCH --ntasks=4                 # Number of tasks (matches num_chains=4)
#SBATCH --time=12:00:00            # Time limit (12 hours)
#SBATCH --partition=shared         # Partition name
#SBATCH --account=phy240043        # Account name
#SBATCH --output=slurm-mcmc-%j.out # Output file

# Load environment
module load anaconda
conda activate tailed-uniform

# Change to project directory
cd /home/x-ctirapongpra/scratch/tailed-uniform-sbi

# Print job information
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "=========================================="

# Run the MCMC script
python scripts/run_mcmc_test_points.py
