from main.simulation import MCMCSimulation
from algorithms import *
import numpy as np
from scipy.stats import multivariate_normal as normal, beta
from target_distributions import *

if __name__ == "__main__":
    dim = 50    # dimension of the target and proposal distributions
    simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.75 ** 2) / (dim ** (1))),  # 2.38**2 / dim
                            num_iterations=20000,
                            algorithm=RandomWalkMH,
                            target_dist=MultimodalDensity(dim, scaling=False),
                            symmetric=True,  # whether to do Metropolis or Metropolis-Hastings: symmetric proposal distribution
                            seed=42)

    chain = simulation.generate_samples()
    print(f"Acceptance rate: {simulation.acceptance_rate():.3f}")
    print(f"Expected squared jump distance: {simulation.expected_squared_jump_distance():.3f}")
    # simulation.autocorrelation_plot()     # Work in progress

    simulation.samples_histogram()
    simulation.traceplot(single_dim=True)   # single_dim=True to plot only the first dimension
    # simulation.traceplot(single_dim=False)  # single_dim=False to plot all dimensions

