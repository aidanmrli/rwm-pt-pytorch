"""
This script runs a single simulation and prints the acceptance rate and the expected squared jump distance.
Useful to study the mixing of the chain in multiple modes by plotting a traceplot and samples histogram of the first dimension.
"""

from main.simulation import MCMCSimulation
from algorithms import *
import numpy as np
from scipy.stats import multivariate_normal as normal, beta
from target_distributions import *

if __name__ == "__main__":
    dim = 20    # dimension of the target and proposal distributions
    # temp_beta_ladder = [1, 0.6785718239036314, 0.43023132454234414, 0.2725213895029845, 0.18473698921221351, 0.01]
    # temp_beta_ladder = [1, 0.7040429390962937, 0.4780498066581084, 0.3336747815093883, 0.2331427383329218, 0.16114482945558686, 0.11392056374013135, 0.07950805093109076, 0.05184317948035542, 0.01]
    temp_beta_ladder = [1., 0.57673998, 0.33056428, 0.18934297, 0.10882526, 0.06198466, 0.03536004, 0.02008734, 0.01]
    simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.38 ** 2) / (dim ** (1))),  # 2.38**2 / dim
                            num_iterations=100000,
                            algorithm=ParallelTemperingRWM,
                            target_dist=MultimodalDensityNew(dim),  # scaling=True for random scaling factors for the components
                            symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                            seed=42,
                            beta_ladder=temp_beta_ladder,
                            swap_acceptance_rate=0.234)

    chain = simulation.generate_samples()
    print(f"Acceptance rate: {simulation.acceptance_rate():.3f}")
    print(f"Expected squared jump distance: {simulation.expected_squared_jump_distance():.3f}")

    simulation.samples_histogram(dim=0)    # plot the histogram of the first dimension
    simulation.traceplot(single_dim=True)   # single_dim=True to plot only the first dimension