#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition="a40,rtx6000,t4v1,t4v2"
#SBATCH --qos=m3
#SBATCH --time=03:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --array=1-10
#SBATCH --output=logs/rwm_%j.out
#SBATCH --error=logs/rwm_%j.err

# Options for GPU: quartet[1-3,5],dgx1,sonata2,bulbasaur,charmander,squirtle
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
# NealFunnel, SuperFunnel

TARGET_DISTRIBUTION="NealFunnel"

# Run the experiment with default parameters but varying seed
python experiment_RWM_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --dim 20 \
    --num_iters 300000 \
    --var_max 3.5 \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 