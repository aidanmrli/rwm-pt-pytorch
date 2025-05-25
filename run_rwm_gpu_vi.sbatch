#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition="a40,rtx6000,t4v1,t4v2"
#SBATCH --qos=m5
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=10G
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
TARGET_DISTRIBUTION="Hypercube"
echo "Starting RWM GPU study for $TARGET_DISTRIBUTION distribution"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the experiment with default parameters but varying seed
python experiment_RWM_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --dim 2 \
    --num_iters 100000 \
    --var_max 2.0 \
    --mode default \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 