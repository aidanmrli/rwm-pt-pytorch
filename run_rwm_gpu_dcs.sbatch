#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=ml
#SBATCH --qos=ml
#SBATCH --account=ml
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --gres=gpu:1
#SBATCH --nodelist=concerto[1-3],overture,quartet[1-3,5],dgx1,sonata2,bulbasaur,charmander,squirtle
#SBATCH --array=1-10
#SBATCH --output=logs/rwm_%j.out
#SBATCH --error=logs/rwm_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source /mfs1/u/$USER/envs/rwm

# Set the seed based on the SLURM array task ID
#### OPTIONS for target distribution: 
# MultivariateNormal, 
# RoughCarpet, RoughCarpetScaled, 
# ThreeMixture, ThreeMixtureScaled, 
# IIDGamma, IIDBeta, Hypercube
TARGET_DISTRIBUTION="MultivariateNormal"
echo "Starting RWM GPU study for $TARGET_DISTRIBUTION distribution"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Seed: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURMD_NODENAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"

# Run the experiment with default parameters but varying seed
python experiment_RWM_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --dim 20 \
    --num_iters 100000 \
    --var_max 3.5 \
    --seed $SLURM_ARRAY_TASK_ID \
    --mode default

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 