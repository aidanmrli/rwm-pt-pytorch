#!/bin/bash

# python3 experiment_RWM.py --dim 5 --var_max 4.0 --target Hypercube --num_iters 1000 --init_seed 0 --num_seeds 5

# python3 experiment_RWM.py --dim 20 --var_max 4.0 --target RoughCarpet --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 30 --var_max 4.0 --target RoughCarpet --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 5 --var_max 4.0 --target RoughCarpet --num_iters 100000 --init_seed 0 --num_seeds 5

# python3 experiment_RWM.py --dim 20 --var_max 4.0 --target RoughCarpetScaled --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 30 --var_max 4.0 --target RoughCarpetScaled --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 50 --var_max 4.0 --target RoughCarpetScaled --num_iters 100000 --init_seed 0 --num_seeds 5

python3 experiment_RWM.py --dim 20 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 30 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 50 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5

python3 experiment_RWM.py --dim 20 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 30 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 50 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
