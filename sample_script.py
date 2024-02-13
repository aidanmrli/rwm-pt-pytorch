from main.simulation import MCMCSimulation
from algorithms import *
import numpy as np
from scipy.stats import multivariate_normal as normal
from target_distributions import *

if __name__ == "__main__":
    dim = 1
    mean_vector = np.zeros(dim)
    # Generate an i.i.d. matrix where all entries are Uniform(0, 1)
    m = np.random.uniform(0, 1, size=(dim, dim)) 
    cov_matrix = np.dot(m, m) + dim * np.eye(dim)   # make the matrix symmetric positive semi-definite
    target_dist = normal(mean_vector, cov_matrix).pdf
    simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.38 ** 2) / dim),
                            num_iterations=1000,
                            algorithm=RandomWalkMH,
                            target_dist=target_dist,
                            symmetric=True,
                            seed=42)

    chain = simulation.generate_samples()
    print(f"Acceptance rate: {simulation.acceptance_rate():.2f}")
    print(f"Expected squared jump distance: {simulation.expected_squared_jump_distance():.2f}")
    simulation.traceplot()
    simulation.autocorrelation_plot()

    if dim == 1:
        simulation.samples_histogram(target_dist)
