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
    swap_acceptance_rates_range = np.linspace(0.01, 0.8, 20)
    num_seeds = 5

    # save results for plotting
    acceptance_rates = []
    expected_squared_jump_distances = []

    # keep scaling factors consistent in the target density across experiments
    target_density_scaled = MultimodalDensityIID(dim, scaling=True)
     # temp_beta_ladder = [1, 0.6785718239036314, 0.43023132454234414, 0.2725213895029845, 0.18473698921221351, 0.01]
    temp_beta_ladder = [1, 0.7040429390962937, 0.4780498066581084, 0.3336747815093883, 0.2331427383329218, 0.16114482945558686, 0.11392056374013135, 0.07950805093109076, 0.05184317948035542, 0.01]


    for a in swap_acceptance_rates_range:
        seed_results_acceptance = []
        seed_results_esjd = []

        for seed_val in range(num_seeds):
            simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.38 ** 2) / (dim ** (1))),  # 2.38**2 / dim
                            num_iterations=50000,
                            algorithm=ParallelTemperingRWM,
                            target_dist=MultimodalDensityNew(dim),  # scaling=True for random scaling factors for the components
                            symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                            seed=seed_val,
                            beta_ladder=None,
                            swap_acceptance_rate=a)
            
            chain = simulation.generate_samples()
            seed_results_acceptance.append(simulation.acceptance_rate())
            seed_results_esjd.append(simulation.expected_squared_jump_distance())

        # calculate average acceptance rate and ESJD for this variance value
        acceptance_rates.append(np.mean(seed_results_acceptance))
        expected_squared_jump_distances.append(np.mean(seed_results_esjd))

    print(f"Maximum ESJD: {max(expected_squared_jump_distances)}")
    print(f"Swap acceptance rate corresponding to maximum ESJD: {acceptance_rates[np.argmax(expected_squared_jump_distances)]}")
    # plot results
    plt.plot(acceptance_rates, expected_squared_jump_distances, label='Expected squared jump distance', marker='x')   
    plt.xlabel('acceptance rate')
    plt.ylabel('ESJD')
    plt.title(f'ESJD vs swap acceptance rate (dim={dim})')
    plt.show()

    # see the last histogram to see if results are consistent
    simulation.samples_histogram(dim=0)  # plot the histogram of the first dimension
    simulation.traceplot(single_dim=True)   # single_dim=True to plot only the first dimension
    # simulation.traceplot(single_dim=False)  # single_dim=False to plot all dimensions

