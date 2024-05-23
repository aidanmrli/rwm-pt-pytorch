#!/bin/bash

# Default values for the arguments
DIM=5
VAR_MAX=2.0
TARGET="Hypercube"
NUM_ITERS=100000
SEED=0
NUM_SEEDS=3

# Parse command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --dim) DIM="$2"; shift ;;
        --var_max) VAR_MAX="$2"; shift ;;
        --target) TARGET="$2"; shift ;;
        --num_iters) NUM_ITERS="$2"; shift ;;
        --init_seed) SEED="$2"; shift ;;
        --num_seeds) NUM_SEEDS="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Run the Python script with the parsed arguments
# python3 experiment_RWM.py --dim "$DIM" --var_max "$VAR_MAX" --target "$TARGET" --num_iters "$NUM_ITERS" --init_seed "$SEED" --num_seeds "$NUM_SEEDS"
python3 experiment_RWM.py --dim 20 --var_max 1.5 --target RoughCarpetScaled --num_iters 100000 --seed 0 --num_seeds 10