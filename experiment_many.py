"""
This script runs many simulations for different variance values and saves the results for plotting.
Useful to study the acceptance rate and the expected squared jump distance for different variance values.
"""

from main.simulation import MCMCSimulation
from algorithms import *
import numpy as np
from scipy.stats import multivariate_normal as normal, beta
from target_distributions import *
import matplotlib.pyplot as plt


if __name__ == "__main__":
    dim = 20    # dimension of the target and proposal distributions
    # run many simulations for different variance values
    var_value_range = np.linspace(0.001, 4, 40)
    num_seeds = 5

    # save results for plotting
    acceptance_rates = []
    expected_squared_jump_distances = []

    # keep scaling factors consistent in the target density across experiments
    target_density_scaled = MultimodalDensityIID(dim, scaling=True)

    for var in var_value_range:
        variance = (var ** 2) / (dim ** (1))
        seed_results_acceptance = []
        seed_results_esjd = []

        for seed_val in range(num_seeds):
            simulation = MCMCSimulation(dim=dim, 
                                        sigma=variance,  # 2.38**2 / dim
                                        num_iterations=100000,
                                        algorithm=RandomWalkMH,
                                        target_dist=target_density_scaled,  # scaling=True for random scaling factors for the components
                                        symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                                        seed=seed_val)
            
            chain = simulation.generate_samples()
            seed_results_acceptance.append(simulation.acceptance_rate())
            seed_results_esjd.append(simulation.expected_squared_jump_distance())

        # calculate average acceptance rate and ESJD for this variance value
        acceptance_rates.append(np.mean(seed_results_acceptance))
        expected_squared_jump_distances.append(np.mean(seed_results_esjd))

    print(f"Maximum ESJD: {max(expected_squared_jump_distances)}")
    print(f"Acceptance rate corresponding to maximum ESJD: {acceptance_rates[np.argmax(expected_squared_jump_distances)]}")
    print(f"Variance value corresponding to maximum ESJD: {var_value_range[np.argmax(expected_squared_jump_distances)]}")
    # plot results
    plt.plot(acceptance_rates, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')   
    plt.xlabel('acceptance rate')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs acceptance rate (dim={dim})')
    plt.show()

    plt.plot(var_value_range, acceptance_rates, label='Acceptance rate', marker='x')  # marker='o'
    plt.xlabel('Variance value (value^2 / dim)')
    plt.ylabel('Acceptance rate')
    plt.title(f'Acceptance rate for different variance values (dim={dim})')
    plt.show()

    plt.plot(var_value_range, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')  # marker='o'
    plt.xlabel('Variance value (value^2 / dim)')
    plt.ylabel('ESJD')
    plt.title(f'ESJD for different variance values (dim={dim})')
    plt.show()

    # see the last histogram to see if results are consistent
    simulation.samples_histogram(dim=0)  # plot the histogram of the first dimension
    simulation.traceplot(single_dim=True)   # single_dim=True to plot only the first dimension

