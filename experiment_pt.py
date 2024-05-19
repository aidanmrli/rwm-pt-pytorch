"""
This parallel tempering script runs many simulations for different 
swap acceptance rates and saves the results for plotting.
Useful to study the acceptance rate and the expected squared jump distance 
for different swap acceptance rate values.
"""

from interfaces import MCMCSimulation
from algorithms import *
import numpy as np
from target_distributions import *
import matplotlib.pyplot as plt
import json


if __name__ == "__main__":
    dim = 20    # dimension of the target and proposal distributions
    ### run many simulations for different variance values
    swap_acceptance_rates_range = np.linspace(0.01, 0.4, 2)
    num_seeds = 3

    ### save results for plotting
    acceptance_rates = []
    expected_squared_jump_distances = []

    ### keep scaling factors consistent in the target density across experiments
    ### set scaling=True for random i.i.d. scaling factors for the components
    ## choose the rough carpet or three mixture or standard multivariate normal
    target_distribution = MultivariateNormal(dim)
    target_distribution = RoughCarpetDistribution(dim, scaling=False)
    # target_distribution = ThreeMixtureDistribution(dim, scaling=False)
    # TODO: WORK IN PROGRESS target_distribution = Hypercube(dim, left_boundary=-1, right_boundary=1)
    # target_distribution = IIDGamma(dim, shape=2, scale=3)
    # target_distribution = IIDBeta(dim, alpha=2, beta=3)

    ### Tune other hyperparameters here
    # algo=ParallelTemperingRWM
    num_iters=1000

    for a in swap_acceptance_rates_range:
        seed_results_acceptance = []
        seed_results_esjd = []

        for seed_val in range(num_seeds):
            simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.38 ** 2) / (dim ** (1))),  # 2.38**2 / dim
                            num_iterations=num_iters,
                            algorithm=ParallelTemperingRWM,
                            target_dist=target_distribution,
                            symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                            seed=seed_val,
                            beta_ladder=None,
                            swap_acceptance_rate=a)
            
            chain = simulation.generate_samples()
            seed_results_acceptance.append(simulation.acceptance_rate())
            seed_results_esjd.append(simulation.pt_expected_squared_jump_distance())

        # calculate average acceptance rate and ESJD
        acceptance_rates.append(np.mean(seed_results_acceptance))
        expected_squared_jump_distances.append(np.mean(seed_results_esjd))

    print(f"Maximum ESJD: {max(expected_squared_jump_distances)}")
    print(f"(Actual) Swap acceptance rate corresponding to maximum ESJD: {acceptance_rates[np.argmax(expected_squared_jump_distances)]}")
    print(f"(Construction) Swap acceptance rate value corresponding to maximum ESJD: {swap_acceptance_rates_range[np.argmax(expected_squared_jump_distances)]}")
    
    ### save the computed ESJDs, acceptance rates, and variances to a file
    data = {
        'expected_squared_jump_distances': expected_squared_jump_distances,
        'acceptance_rates': acceptance_rates,
        'swap_acceptance_rates_range': swap_acceptance_rates_range
    }
    with open(f"data/{target_distribution.get_name()}_PTrwm_dim{dim}_{num_iters}iters.json", "w") as file:
        json.dump(data, file)

    ### plot results
    ### with the actual swap acceptance rate of the simulation
    plt.plot(acceptance_rates, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')   
    plt.xlabel('swap acceptance rate (actual)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs swap acceptance rate (dim={dim})')
    filename = f"images/ESJDvsSwapAcceptActual_{target_distribution.get_name()}_PTrwm_dim{dim}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()
    # plt.show()

    ### with the desired swap acceptance rate when the ladder was constructed
    plt.plot(swap_acceptance_rates_range, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')   
    plt.xlabel('swap acceptance rate (construction)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs swap acceptance rate (dim={dim})')
    filename = f"images/ESJDvsSwapAcceptConstr_{target_distribution.get_name()}_PTrwm_dim{dim}_{num_iters}iters"
    plt.savefig(filename)
    plt.clf()
    # plt.show()

    ### see the last histogram to see if results are consistent
    simulation.samples_histogram(axis=0)  # plot the histogram of the first dimension
    simulation.traceplot(single_dim=True)   # single_dim=True to plot only the first dimension

