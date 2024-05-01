import numpy as np
from scipy.stats import multivariate_normal as normal
from main import MHAlgorithm
from typing import Callable
from main import TargetDistribution

class RandomWalkMH(MHAlgorithm):
    """Implementation of the Random Walk Metropolis-Hastings algorithm for sampling from a target distribution."""
    def __init__(self, dim, var, target_dist: TargetDistribution = None, symmetric=True, beta=1.0, beta_ladder=None, swap_acceptance_rate=None):
        """Initialize the RandomWalkMH algorithm. Note: the beta_ladder and swap_acceptance_rate are not used in this implementation,
        this is due to higher-level code that uses the same interface for different algorithms."""
        super().__init__(dim, var, target_dist, symmetric)
        self.num_acceptances = 0    # use this to calculate acceptance rate
        self.acceptance_rate = 0
        self.log_target_density_curr_state = -np.inf    # this is the log density of the current state, used to reduce redundant computation
        self.beta = beta

    def step(self):
        """Take a step using the Random Walk Metropolis-Hastings algorithm.
        Add the new state to the chain with probability min(1, A) where A is the acceptance probability.
        """
        # np.eye(self.dim) * (self.var) is the covariance matrix, 
        proposed_state = np.random.multivariate_normal(self.chain[-1], (self.var / self.beta) * np.eye(self.dim))
    
        log_accept_ratio, log_target_density_proposed_state = self.log_accept_prob(proposed_state, self.log_target_density_curr_state, self.chain[-1])
        # accept the proposed state with probability min(1, A)
        if log_accept_ratio > 0 or np.random.random() < np.exp(log_accept_ratio):
            self.chain.append(proposed_state)
            self.log_target_density_curr_state = log_target_density_proposed_state
            self.num_acceptances += 1
            self.acceptance_rate = self.num_acceptances / len(self.chain)
        else:
            self.chain.append(self.chain[-1])
            self.acceptance_rate = self.num_acceptances / len(self.chain)

    def log_accept_prob(self, proposed_state, log_target_density_curr_state, current_state):
        """Calculate the log acceptance probability for the proposed state given the current state.
        We use the log density of the target distribution for numerical stability in high dimensions.

        Args:
            proposed_state (np.ndarray): The proposed state.
            current_state (np.ndarray): The current state.
            symmetric (bool): Whether the proposal distribution is symmetric. Default is False.
            
        Returns:
            float: The log acceptance probability."""
        if self.symmetric:
            log_target_density_proposed_state = np.log(self.target_density(proposed_state) + 1e-300) # for numerical stability
            return self.beta * (log_target_density_proposed_state - log_target_density_curr_state), log_target_density_proposed_state
        else:
            log_target_density_proposed_state = np.log(self.target_density(proposed_state))
            log_target_term = self.beta * (log_target_density_proposed_state - log_target_density_curr_state)
            log_proposal_term = (np.log(normal.pdf(current_state, mean=proposed_state, 
                                                   cov=np.eye(self.dim) * (self.var))) 
                                - np.log(normal.pdf(proposed_state, mean=current_state, 
                                                    cov=np.eye(self.dim) * (self.var))))

            return log_target_term + log_proposal_term, log_target_density_proposed_state
