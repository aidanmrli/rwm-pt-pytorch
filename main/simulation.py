import numpy as np
import matplotlib.pyplot as plt
from .metropolis import MHAlgorithm
from algorithms import *
from typing import Optional, Callable

class MCMCSimulation:
    def __init__(self, 
                 dim: int, 
                 sigma: float, 
                 num_iterations: int = 1000, 
                 algorithm: MHAlgorithm = RandomWalkMH,
                 target_dist: Callable = None,
                 symmetric: bool = True,
                 seed: Optional[int] = None):
        self.num_iterations = num_iterations
        self.algorithm = algorithm(dim, sigma, target_dist, symmetric)
        if seed:
            np.random.seed(seed)
    
    def reset(self):
        """Reset the simulation to the initial state."""
        self.algorithm.reset()

    def has_run(self):
        """Return whether the algorithm has been run."""
        return len(self.algorithm.chain) > 1

    def generate_samples(self):
        if self.has_run():
            raise ValueError("Please reset the algorithm before running it again.")
        for _ in range(self.num_iterations):
            self.algorithm.step()
        return self.algorithm.chain
    
    def acceptance_rate(self):
        """Return the acceptance rate of the algorithm."""
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        return self.algorithm.acceptance_rate

    def expected_squared_jump_distance(self, n: Optional[int] = 1):
        """Calculate the expected squared jump distance for the 
        last n elements of the Markov chain. 
        Args:
            n (int): The number of steps to take in the chain.
        Returns:
            float: The expected squared jump distance.
        """
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        chain = np.array(self.algorithm.chain)
        return np.mean(np.sum((chain[:-n] - chain[:-1]) ** 2, axis=1))

    def traceplot(self):
        """Visualize the traceplot of the Markov chain.
        The traceplot plots the values of the parameters 
        against the iteration number in the Markov chain.
        """
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        
        chain = np.array(self.algorithm.chain)
        for i in range(self.algorithm.dim):
            plt.plot(chain[:, i], label=f"Dimension {i + 1}", alpha=0.7, lw=0.5)

        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.show()

    
    def autocorrelation_plot(self):
        """Visualize the autocorrelation of the Markov chain.
        """
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        chain = np.array(self.algorithm.chain)
        autocorr = np.zeros(self.algorithm.dim)
        for i in range(self.algorithm.dim):
            autocorr[i] = np.correlate(chain[:, i] - np.mean(chain[:, i]), chain[:, i] - np.mean(chain[:, i]), mode='full')[chain.shape[0] - 1]
        
        plt.stem(range(len(autocorr)), autocorr)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()