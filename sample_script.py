from main.simulation import MCMCSimulation
from algorithms import *
import numpy as np
from scipy.stats import multivariate_normal as normal

if __name__ == "__main__":
    dim = 5
    simulation = MCMCSimulation(dim=dim, 
                            sigma=((2.38 ** 2) / dim),
                            num_iterations=1000,
                            algorithm=RandomWalkMH,
                            target_dist=normal(np.zeros(dim), np.eye(dim)).pdf,
                            symmetric=True,
                            seed=42)

    chain = simulation.generate_samples()
    simulation.traceplot()
    simulation.autocorrelation_plot()
