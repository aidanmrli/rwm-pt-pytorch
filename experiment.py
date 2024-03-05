from main.simulation import MCMCSimulation
from algorithms import *
import numpy as np
from scipy.stats import multivariate_normal as normal, beta
from target_distributions import *

if __name__ == "__main__":
    dim = 50
    # mean_vector = np.ones(dim)
    # # Generate an i.i.d. matrix where all entries are Uniform(0, 1)
    # m = np.random.uniform(0, 1, size=(dim, dim)) 
    # cov_matrix = np.dot(m, m) + dim * np.eye(dim)   # make the matrix symmetric positive semi-definite
    # target_dist = normal(mean_vector, cov_matrix).pdf

    # alpha = 2  # Shape parameter alpha
    # beta_param = 5  # Shape parameter beta
    # target_dist = beta(alpha, beta_param).pdf

    # target_dist = iid_product_density
    target_dist = MultimodalDensity(dim, scaling=False)

    simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.75 ** 2) / (dim ** (1))),  # 2.38**2 / dim
                            num_iterations=20000,
                            algorithm=RandomWalkMH,
                            target_dist=target_dist,
                            symmetric=True,
                            seed=42)

    chain = simulation.generate_samples()
    print(f"Acceptance rate: {simulation.acceptance_rate():.3f}")
    print(f"Expected squared jump distance: {simulation.expected_squared_jump_distance():.3f}")
    # simulation.traceplot(single_dim=False)  # single_dim=False to plot all dimensions
    # simulation.autocorrelation_plot()

    simulation.samples_histogram()
    simulation.traceplot(single_dim=True)   # single_dim=True to plot only the first dimension
