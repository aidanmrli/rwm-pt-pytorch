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
#SBATCH --array=1-20%10
#SBATCH --output=logs/rwm_%j.out
#SBATCH --error=logs/rwm_%j.err

#concerto[1-3],overture,quartet[1-3,5],dgx1,sonata2,
# Options for GPU: 
# concerto[1-3],overture
# quartet[1-3,5],dgx1,sonata2,bulbasaur,charmander,squirtle
# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
. /mfs1/u/$USER/envs/rwm

#### OPTIONS for target distribution: 
# MultivariateNormal, MultivariateNormalScaled
# RoughCarpet, RoughCarpetScaled, 
# ThreeMixture, ThreeMixtureScaled, 
# IIDGamma, IIDBeta, Hypercube
# FullRosenbrock, EvenRosenbrock, HybridRosenbrock
# NealFunnel, SuperFunnel

TARGET_DISTRIBUTION="MultivariateNormal"

#### OPTIONS for proposal distribution:
# Normal, Laplace, UniformRadius
PROPOSAL_DISTRIBUTION="Laplace" # Default proposal

python experiment_RWM_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --proposal $PROPOSAL_DISTRIBUTION \
    --dim 2 \
    --num_iters 200000 \
    --var_max 4 \
    --burn_in 1000 \
    --hybrid_rosenbrock_n1 3 \
    --hybrid_rosenbrock_n2 2 \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 