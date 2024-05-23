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
    elif name == "Hypercube":
        return Hypercube(dim, left_boundary=-1, right_boundary=1)
    elif name == "IIDGamma":
        return IIDGamma(dim, shape=2, scale=3)
    elif name == "IIDBeta":
        return IIDBeta(dim, alpha=2, beta=3)
    else:
        raise ValueError("Unknown target distribution name")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RWM simulations with various parameters")
    parser.add_argument("--dim", type=int, default=5, help="Dimension of the target and proposal distributions")
    parser.add_argument("--var_max", type=float, default=2.0, help="Upper bound of the variance value range")
    parser.add_argument("--target", type=str, default="Hypercube", help="Target density distribution")
    parser.add_argument("--num_iters", type=int, default=100000, help="Number of iterations for each MCMC simulation")
    parser.add_argument("--init_seed", type=int, default=0, help="Starting seed value")
    parser.add_argument("--num_seeds", type=int, default=3, help="Number of seeds to use in the simulations")

    args = parser.parse_args()

    dim = args.dim
    var_value_range = np.linspace(0.000001, args.var_max, 40)
    num_seeds = args.num_seeds
    num_iters = args.num_iters
    target_distribution = get_target_distribution(args.target, dim)
    
    acceptance_rates = []
    expected_squared_jump_distances = []

    for i in range(len(var_value_range)):
        var = var_value_range[i]
        print(f"Variance {i + 1} out of {len(var_value_range)}")
        variance = (var ** 2) / (dim ** (1))
        seed_results_acceptance = []
        seed_results_esjd = []

        for seed_val in range(args.init_seed, args.init_seed + num_seeds):
            simulation = MCMCSimulation(dim=dim, 
                                        sigma=variance,
                                        num_iterations=num_iters,
                                        algorithm=RandomWalkMH,
                                        target_dist=target_distribution,
                                        symmetric=True,
                                        seed=seed_val)
            
            chain = simulation.generate_samples()
            seed_results_acceptance.append(simulation.acceptance_rate())
            seed_results_esjd.append(simulation.expected_squared_jump_distance())

        acceptance_rates.append(np.mean(seed_results_acceptance))
        expected_squared_jump_distances.append(np.mean(seed_results_esjd))

    max_esjd = max(expected_squared_jump_distances)
    max_esjd_index = np.argmax(expected_squared_jump_distances)
    max_acceptance_rate = acceptance_rates[max_esjd_index]
    max_variance_value = var_value_range[max_esjd_index]
    print(f"Maximum ESJD: {max_esjd}")
    print(f"Acceptance rate corresponding to maximum ESJD: {max_acceptance_rate}")
    print(f"Variance value corresponding to maximum ESJD: {max_variance_value}")

    data = {
        'expected_squared_jump_distances': expected_squared_jump_distances,
        'acceptance_rates': acceptance_rates,
        'var_value_range': var_value_range.tolist(),
        'max_esjd': max_esjd,
        'max_acceptance_rate': max_acceptance_rate,
        'max_variance_value': max_variance_value
    }
    with open(f"data/{target_distribution.get_name()}_RWM_dim{dim}_{num_iters}iters.json", "w") as file:
        json.dump(data, file)

    plt.plot(acceptance_rates, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')   
    plt.xlabel('acceptance rate')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs acceptance rate (dim={dim})')
    filename = f"images/ESJDvsAccept_{target_distribution.get_name()}_RWM_dim{dim}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()

    plt.plot(var_value_range, acceptance_rates, label='Acceptance rate', marker='x')
    plt.xlabel('Variance value (value^2 / dim)')
    plt.ylabel('Acceptance rate')
    plt.title(f'Acceptance rate for different variance values (dim={dim})')
    filename = f"images/AcceptvsVar_{target_distribution.get_name()}_RWM_dim{dim}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()

    plt.plot(var_value_range, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')
    plt.xlabel('Variance value (value^2 / dim)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD for different variance values (dim={dim})')
    filename = f"images/ESJDvsVar_{target_distribution.get_name()}_RWM_dim{dim}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()

    simulation.samples_histogram(axis=0)
    simulation.traceplot(single_dim=True)
