#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=REPLACE_WITH_PARTITION
#SBATCH --qos=REPLACE_WITH_QOS
#SBATCH --nodelist=REPLACE_WITH_NODELIST
#SBATCH --time=07:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --array=1-20
#SBATCH --output=logs/pt_%j.out
#SBATCH --error=logs/pt_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source venv/bin/activate # CHANGE TO YOUR VENV NAME

#### Options for target distribution: 
# MultivariateNormal, 
# RoughCarpet, RoughCarpetScaled, 
# ThreeMixture, ThreeMixtureScaled, 
# IIDGamma, IIDBeta, Hypercube
# FullRosenbrock, EvenRosenbrock, HybridRosenbrock
TARGET_DISTRIBUTION="ThreeMixture"
DIM=10

python experiment_pt_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --dim $DIM \
    --num_iters 200000 \
    --swap_accept_max 0.5 \
    --burn_in 1000 \
    --N_samples_swap_est 1000000 \
    --iterative_tolerance 0.0001 \
    --iterative_max_pn_steps 1000 \
    --iterative_fail_tol_factor 1 \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed PT study for target $TARGET_DISTRIBUTION, dim $DIM, seed $SLURM_ARRAY_TASK_ID" 