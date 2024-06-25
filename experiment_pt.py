import argparse
from interfaces import MCMCSimulation
from algorithms import *
import numpy as np
from target_distributions import *
import matplotlib.pyplot as plt
import json

def get_target_distribution(name, dim):
    if name == "MultivariateNormal":
        return MultivariateNormal(dim)
    elif name == "RoughCarpet":
        return RoughCarpetDistribution(dim, scaling=False)
    elif name == "ThreeMixture":
        return ThreeMixtureDistribution(dim, scaling=False)
    else:
        raise ValueError("Unknown target distribution name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Parallel Tempering simulations with various parameters")
    parser.add_argument("--dim", type=int, default=20, help="Dimension of the target and proposal distributions")
    parser.add_argument("--swap_accept_max", type=float, default=0.6, help="Upper bound of the swap acceptance rate range")
    parser.add_argument("--target", type=str, default="RoughCarpet", help="Target density distribution")
    parser.add_argument("--num_iters", type=int, default=100000, help="Number of iterations for the MCMC simulation")
    parser.add_argument("--init_seed", type=int, default=0, help="Starting seed value")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to use in the simulations")

    args = parser.parse_args()

    dim = args.dim
    swap_acceptance_rates_range = np.linspace(0.01, args.swap_accept_max, 40)
    num_seeds = args.num_seeds
    num_iters = args.num_iters
    target_distribution = get_target_distribution(args.target, dim)

    acceptance_rates = []
    expected_squared_jump_distances = []

    for i in range(len(swap_acceptance_rates_range)):
        a = swap_acceptance_rates_range[i]
        print(f"{target_distribution.get_name()} dim{dim}: Temperature spacing {i + 1} out of {len(swap_acceptance_rates_range)}")
        seed_results_acceptance = []
        seed_results_esjd = []
        constructed_beta_ladder = None

        for seed_val in range(args.init_seed, args.init_seed + num_seeds):
            simulation = MCMCSimulation(dim=dim, 
                                        sigma=((2.38 ** 2) / (dim ** (1))),  # 2.38**2 / dim
                                        num_iterations=num_iters,
                                        algorithm=ParallelTemperingRWM,
                                        target_dist=target_distribution,
                                        symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                                        seed=seed_val,
                                        beta_ladder=constructed_beta_ladder,
                                        swap_acceptance_rate=a)
            constructed_beta_ladder = simulation.algorithm.beta_ladder
            
            chain = simulation.generate_samples()
            seed_results_acceptance.append(simulation.acceptance_rate())
            seed_results_esjd.append(simulation.pt_expected_squared_jump_distance())

        acceptance_rates.append(np.mean(seed_results_acceptance))
        expected_squared_jump_distances.append(np.mean(seed_results_esjd))

    max_esjd = max(expected_squared_jump_distances)
    max_esjd_index = np.argmax(expected_squared_jump_distances)
    max_actual_acceptance_rate = acceptance_rates[max_esjd_index]
    max_constr_acceptance_rate = swap_acceptance_rates_range[max_esjd_index]

    print(f"{target_distribution.get_name()} dim{dim} Maximum ESJD: {max_esjd}")
    print(f"{target_distribution.get_name()} dim{dim} (Actual) Swap acceptance rate corresponding to maximum ESJD: {max_actual_acceptance_rate}")
    print(f"{target_distribution.get_name()} dim{dim} (Construction) Swap acceptance rate value corresponding to maximum ESJD: {max_constr_acceptance_rate}")

    data = {
        'max_esjd': max_esjd,
        'max_actual_acceptance_rate': max_actual_acceptance_rate,
        'max_constr_acceptance_rate': max_constr_acceptance_rate,
        'expected_squared_jump_distances': expected_squared_jump_distances,
        'acceptance_rates': acceptance_rates,
        'swap_acceptance_rates_range': swap_acceptance_rates_range.tolist()
    }
    with open(f"data/{target_distribution.get_name()}_PTrwm_dim{dim}_seed{args.init_seed}_{num_iters}iters.json", "w") as file:
        json.dump(data, file)

    plt.plot(acceptance_rates, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')   
    plt.axvline(x=0.234, color='red', linestyle=':', label='x = 0.234')
    plt.text(0.234, plt.gca().get_ylim()[0], '0.234', ha='center', va='top')
    plt.xlabel('swap acceptance rate (actual)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs swap acceptance rate (dim={dim})')
    filename = f"images/ESJDvsSwapAcceptActual_{target_distribution.get_name()}_PTrwm_dim{dim}_seed{args.init_seed}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()

    plt.plot(swap_acceptance_rates_range, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')
    plt.axvline(x=0.234, color='red', linestyle=':', label='x = 0.234')
    plt.text(0.234, plt.gca().get_ylim()[0], '0.234', ha='center', va='top')
    plt.xlabel('swap acceptance rate (construction)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs swap acceptance rate (dim={dim})')
    filename = f"images/ESJDvsSwapAcceptConstr_{target_distribution.get_name()}_PTrwm_dim{dim}_seed{args.init_seed}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()
