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
#SBATCH --nodelist=concerto[1-3],overture,quartet[1-3,5],dgx1,sonata2
#SBATCH --array=1-30
#SBATCH --output=logs/rwm_%j.out
#SBATCH --error=logs/rwm_%j.err

#
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
TARGET_DISTRIBUTION="ThreeMixtureScaled"

python experiment_RWM_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --dim 50 \
    --num_iters 300000 \
    --var_max 3.5 \
    --hybrid_rosenbrock_n1 3 \
    --hybrid_rosenbrock_n2 2 \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 