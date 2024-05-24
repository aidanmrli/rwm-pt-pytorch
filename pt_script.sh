#!/bin/bash

python3 experiment_pt.py --dim 20 --swap_accept_max 0.6 --target MultivariateNormal --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_pt.py --dim 20 --swap_accept_max 0.6 --target RoughCarpet --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_pt.py --dim 30 --swap_accept_max 0.6 --target RoughCarpet --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_pt.py --dim 20 --swap_accept_max 0.6 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_pt.py --dim 30 --swap_accept_max 0.6 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5