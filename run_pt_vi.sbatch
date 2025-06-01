#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition="a40,rtx6000,t4v1,t4v2"
#SBATCH --qos=m2
#SBATCH --time=07:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --array=1-20
#SBATCH --output=logs/pt_vi_%j.out
#SBATCH --error=logs/pt_vi_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source /h/liaidan/random-walk-metropolis/venv/bin/activate

# Set the seed based on the SLURM array task ID
#### OPTIONS for target distribution: 
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
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed PT study on VI for target $TARGET_DISTRIBUTION, dim $DIM, seed $SLURM_ARRAY_TASK_ID" 