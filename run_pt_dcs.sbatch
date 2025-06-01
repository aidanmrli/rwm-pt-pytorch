#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=quartet[1-3,5],dgx1,sonata2,bulbasaur,charmander,squirtle
#SBATCH --array=1-20%10
#SBATCH --output=logs/pt_dcs_%j.out
#SBATCH --error=logs/pt_dcs_%j.err

# Options for GPU: 
# concerto[1-3],overture
# quartet[1-3,5],dgx1,sonata2,bulbasaur,charmander,squirtle
# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
. /mfs1/u/$USER/envs/rwm

# Set the seed based on the SLURM array task ID
#### OPTIONS for target distribution: 
# MultivariateNormal, 
# RoughCarpet, RoughCarpetScaled, 
# ThreeMixture, ThreeMixtureScaled, 
# IIDGamma, IIDBeta, Hypercube
# FullRosenbrock, EvenRosenbrock, HybridRosenbrock
TARGET_DISTRIBUTION="RoughCarpet"

python experiment_pt_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --dim 20 \
    --num_iters 200000 \
    --swap_accept_max 0.5 \
    --burn_in 1000 \
    --N_samples_swap_est 1000000 \
    --iterative_tolerance 0.0001 \
    --iterative_max_pn_steps 1000 \
    --iterative_fail_tol_factor 1 \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 