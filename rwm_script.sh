#!/bin/bash
python single_run_experiment.py --target IIDGamma --dim 2 --normal_base_variance 28.0 --num_iters 100000
python single_run_experiment.py --target IIDGamma --dim 5 --normal_base_variance 18.0 --num_iters 100000
python single_run_experiment.py --target IIDGamma --dim 10 --normal_base_variance 14.0 --num_iters 100000
python single_run_experiment.py --target IIDGamma --dim 30 --normal_base_variance 11.0 --num_iters 100000
python single_run_experiment.py --target IIDGamma --dim 50 --normal_base_variance 10.0 --num_iters 100000
python single_run_experiment.py --target IIDGamma --dim 100 --normal_base_variance 8.5 --num_iters 100000


# python3 experiment_RWM.py --dim 2 --var_max 0.9 --target IIDBeta --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 5 --var_max 0.9 --target IIDBeta --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 10 --var_max 0.9 --target IIDBeta --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 20 --var_max 0.9 --target IIDBeta --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 30 --var_max 0.9 --target IIDBeta --num_iters 100000 --init_seed 0 --num_seeds 5
python3 experiment_RWM.py --dim 50 --var_max 0.9 --target IIDBeta --num_iters 100000 --init_seed 0 --num_seeds 5

# python3 experiment_RWM.py --dim 2 --var_max 3.0 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 5 --var_max 2.0 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 10 --var_max 2.0 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 20 --var_max 1.5 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 30 --var_max 1.5 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 50 --var_max 1.2 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 50 --var_max 1.0 --target Hypercube --num_iters 100000 --init_seed 0 --num_seeds 5

python single_run_experiment.py --target RoughCarpet --dim 2 --normal_base_variance 24.0 --num_iters 100000 --burn_in 1000
python single_run_experiment.py --target RoughCarpet --dim 5 --normal_base_variance 14.0 --num_iters 100000 --burn_in 1000
python single_run_experiment.py --target RoughCarpet --dim 10 --normal_base_variance 5.7 --num_iters 100000 --burn_in 1000
python single_run_experiment.py --target RoughCarpet --dim 20 --normal_base_variance 5.0 --num_iters 100000 --burn_in 1000
python single_run_experiment.py --target RoughCarpet --dim 30 --normal_base_variance 4.7 --num_iters 100000 --burn_in 1000
python single_run_experiment.py --target RoughCarpet --dim 50 --normal_base_variance 4.7 --num_iters 100000 --burn_in 1000

# python3 experiment_RWM.py --dim 2 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 5 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 10 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
# # python3 experiment_RWM.py --dim 20 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
# # python3 experiment_RWM.py --dim 30 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5
# # python3 experiment_RWM.py --dim 50 --var_max 4.0 --target ThreeMixture --num_iters 100000 --init_seed 0 --num_seeds 5

# python3 experiment_RWM.py --dim 2 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 5 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
# python3 experiment_RWM.py --dim 10 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
# # python3 experiment_RWM.py --dim 20 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
# # python3 experiment_RWM.py --dim 30 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5
# # python3 experiment_RWM.py --dim 50 --var_max 4.0 --target ThreeMixtureScaled --num_iters 100000 --init_seed 0 --num_seeds 5

python experiment_RWM_GPU.py --dim 30 --var_max 1.5 --target FullRosenbrock --num_iters 100000

python single_run_experiment.py --target FullRosenbrock --dim 2 --normal_base_variance 2.8 --num_iters 100000
python single_run_experiment.py --target FullRosenbrock --dim 5 --normal_base_variance 1.5 --num_iters 100000
python single_run_experiment.py --target FullRosenbrock --dim 10 --normal_base_variance 1.3 --num_iters 100000
python single_run_experiment.py --target FullRosenbrock --dim 20 --normal_base_variance 1.3 --num_iters 100000
python single_run_experiment.py --target FullRosenbrock --dim 30 --normal_base_variance 1.3 --num_iters 100000

python single_run_experiment.py --target EvenRosenbrock --dim 2 --normal_base_variance 2.8 --num_iters 100000
python single_run_experiment.py --target EvenRosenbrock --dim 4 --normal_base_variance 1.3 --num_iters 100000
python single_run_experiment.py --target EvenRosenbrock --dim 10 --normal_base_variance 0.6 --num_iters 100000
python single_run_experiment.py --target EvenRosenbrock --dim 20 --normal_base_variance 0.6 --num_iters 100000
python single_run_experiment.py --target EvenRosenbrock --dim 30 --normal_base_variance 0.6 --num_iters 100000

python single_run_experiment.py --target HybridRosenbrock --dim 3 --normal_base_variance 1.6 --num_iters 100000 --hybrid_rosenbrock_n1 2 --hybrid_rosenbrock_n2 2
python single_run_experiment.py --target HybridRosenbrock --dim 5 --normal_base_variance 1.4 --num_iters 100000 --hybrid_rosenbrock_n1 3 --hybrid_rosenbrock_n2 2
python single_run_experiment.py --target HybridRosenbrock --dim 9 --normal_base_variance 1.2 --num_iters 100000 --hybrid_rosenbrock_n1 5 --hybrid_rosenbrock_n2 2
python single_run_experiment.py --target HybridRosenbrock --dim 19 --normal_base_variance 1.2 --num_iters 100000 --hybrid_rosenbrock_n1 7 --hybrid_rosenbrock_n2 3
python single_run_experiment.py --target HybridRosenbrock --dim 29 --normal_base_variance 1.2 --num_iters 100000 --hybrid_rosenbrock_n1 8 --hybrid_rosenbrock_n2 4

python single_run_experiment.py --target NealFunnel --dim 5 --normal_base_variance 9.0 --num_iters 100000 --burn_in 50000
python single_run_experiment.py --target NealFunnel --dim 10 --normal_base_variance 8.0 --num_iters 100000 --burn_in 50000
python single_run_experiment.py --target NealFunnel --dim 20 --normal_base_variance 6.6 --num_iters 100000 --burn_in 50000
python single_run_experiment.py --target NealFunnel --dim 30 --normal_base_variance 4.8 --num_iters 100000 --burn_in 50000
