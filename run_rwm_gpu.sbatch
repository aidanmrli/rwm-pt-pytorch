#!/bin/bash
#SBATCH --job-name=bash
#SBATCH --partition=REPLACE_WITH_PARTITION
#SBATCH --qos=REPLACE_WITH_QOS
#SBATCH --account=REPLACE_WITH_ACCOUNT
#SBATCH --nodelist=REPLACE_WITH_NODELIST
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --array=1-20
#SBATCH --mem=20G
#SBATCH --gres=gpu:1
#SBATCH --time=00:59:00
#SBATCH --output=logs/rwm_%j.out
#SBATCH --error=logs/rwm_%j.err

# Create logs directory if it doesn't exist
mkdir -p logs

# Activate the environment
source /venv/bin/activate # CHANGE TO YOUR VENV NAME

#### Options for target distribution: 
# MultivariateNormal, MultivariateNormalScaled
# RoughCarpet, RoughCarpetScaled, 
# ThreeMixture, ThreeMixtureScaled, 
# IIDGamma, IIDBeta, Hypercube
# FullRosenbrock, EvenRosenbrock, HybridRosenbrock
# NealFunnel, SuperFunnel
TARGET_DISTRIBUTION="RoughCarpetScaled"

#### Options for proposal distribution:
# Normal, Laplace, UniformRadius
PROPOSAL_DISTRIBUTION="Normal" # Default proposal

python experiment_RWM_GPU.py \
    --target $TARGET_DISTRIBUTION \
    --proposal $PROPOSAL_DISTRIBUTION \
    --dim 20 \
    --num_iters 200000 \
    --var_max 4.0 \
    --burn_in 1000 \
    --hybrid_rosenbrock_n1 8 \
    --hybrid_rosenbrock_n2 4 \
    --seed $SLURM_ARRAY_TASK_ID

echo "Completed RWM GPU study for seed $SLURM_ARRAY_TASK_ID" 