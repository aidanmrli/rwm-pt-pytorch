"""
This script runs a single simulation of either standard RWM or 
parallel tempering and prints the acceptance rate 
and the expected squared jump distance.
Useful to study the mixing of the chain in multiple modes by plotting 
a traceplot and samples histogram of the first dimension.
"""

from interfaces import MCMCSimulation
from algorithms import *
import numpy as np
from target_distributions import *

if __name__ == "__main__":
    dim = 20    # dimension of the target and proposal distributions
    
    ### Reuse some beta ladder if you would like to save time
    # temp_beta_ladder = [1, 0.7201258143345616, 0.5351535500164732, 0.40699808631201695, 0.31168319620186075, 0.23789880207916622, 0.1726209665306218, 0.117096477032898, 0.07923549613447455, 0.01]
    
    ### choose the rough carpet or three mixture or standard multivariate normal
    # target_distribution = MultivariateNormal(dim)
    # target_distribution = RoughCarpetDistribution(dim, scaling=False)
    # target_distribution = ThreeMixtureDistribution(dim, scaling=False)
    # target_distribution = Hypercube(dim, left_boundary=-1, right_boundary=1)
    # target_distribution = IIDGamma(dim, shape=2, scale=3)
    target_distribution = IIDBeta(dim, alpha=2, beta=5)

    simulation = MCMCSimulation(dim=dim, 
                            sigma=((0.5) / (dim ** (1))),  # 2.38**2 / dim
                            num_iterations=100000,
                            algorithm=RandomWalkMH, # RandomWalkMH or ParallelTemperingRWM
                            target_dist=target_distribution,
                            symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                            seed=1,
                            beta_ladder=None,
                            swap_acceptance_rate=0.234)

    chain = simulation.generate_samples()
    print(f"Acceptance rate: {simulation.acceptance_rate():.3f}")
    print(f"Expected squared jump distance: {simulation.expected_squared_jump_distance():.3f}")

    simulation.samples_histogram(axis=0, show=True)    # plot the histogram of the first dimension
    simulation.traceplot(single_dim=True, show=True)   # single_dim=True to plot only the first dimension