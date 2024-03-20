import numpy as np
from scipy.stats import multivariate_normal as normal
from main import MHAlgorithm
from typing import Callable

class ParallelTemperingRWM(MHAlgorithm):
    """Implementation of the Random Walk Metropolis-Hastings with Parallel Tempering algorithm for sampling from a target distribution."""
    def __init__(
            self, 
            dim, 
            var, 
            target_dist: Callable = None, 
            symmetric=True,
            temp_ladder=None,
            geom_temp_spacing=True
            ):
        super().__init__(dim, var, target_dist, symmetric)
        self.num_swap_attempts = 0
        self.num_acceptances = 0    # use this to calculate acceptance rate
        self.acceptance_rate = 0    # this refers to SWAP acceptance rate
        self.chains = []
        self.temp_ladder = temp_ladder
        if self.temp_ladder is None:
            self.temp_ladder = []
            if geom_temp_spacing:
                # construct a geometrically spaced temperature ladder
                if geom_temp_spacing:
                    beta_0, beta_min = 1, 1e-2
                    curr_beta = beta_0
                    c = 0.5     # set the geometric spacing constant
                    while curr_beta > beta_min:
                        self.temp_ladder.append(curr_beta)
                        self.chains.append(self.chain.copy())  # initialize one chain for each temperature
                        curr_beta = curr_beta * c
                    
                    self.temp_ladder.append(beta_min)
                    self.chains.append(self.chain.copy())

                # iteratively construct the inverse temperatures using simulation-based approach
                else:
                    # TODO: implement the simulation-based approach to construct spacing of prob 0.23
                    # also consider adaptive temperature ladder
                    pass
        self.chain = self.chains[0]  # the first chain is the "cold" chain


    def attempt_swap(self, j, k):
        """Attempt to swap states between two chains based on Metropolis criteria."""
        swap_prob = min(1, np.exp(self.log_swap_prob(j, k)))
        self.num_swap_attempts += 1
        if np.random.random() < swap_prob:
            # swap the states
            temp = self.chains[k][-1].copy()
            self.chains[k][-1] = self.chains[j][-1].copy()
            self.chains[j][-1] = temp
            self.num_acceptances += 1
            self.acceptance_rate = self.num_acceptances / self.num_swap_attempts


    def log_swap_prob(self, j, k):
        """Calculate the log probability of swapping states between two chains."""
        log_prob = (
            self.temp_ladder[j] * np.log(self.target_dist(self.chains[k][-1])) + 
            self.temp_ladder[k] * np.log(self.target_dist(self.chains[j][-1])) -
            self.temp_ladder[j] * np.log(self.target_dist(self.chains[j][-1])) -
            self.temp_ladder[k] * np.log(self.target_dist(self.chains[k][-1]))
                )
        return log_prob
    
    def step(self, swap=False):
        """Take a step using the Random Walk Metropolis-Hastings algorithm.
        Add the new state to the chain with probability min(1, A) where A is the acceptance probability.
        """
        for i in range(len(self.chains)):
            curr_chain = self.chains[i]
            if swap and i < len(self.chains) - 1:
                self.attempt_swap(i, i+1)   # this handles swap acceptance rate as well
            else:
                proposed_state = np.random.multivariate_normal(curr_chain[-1], np.eye(self.dim) * (self.var / self.temp_ladder[i]))
                log_accept_ratio = self.log_accept_prob(proposed_state, curr_chain[-1])

                # accept the proposed state with probability min(1, A)
                if log_accept_ratio > 0 or np.random.random() < np.exp(log_accept_ratio):
                    curr_chain.append(proposed_state)
                else:
                    curr_chain.append(curr_chain[-1])


    def log_accept_prob(self, proposed_state, current_state):
        """Calculate the log acceptance probability for the proposed state given the current state.
        We use the log density of the target distribution for numerical stability in high dimensions.

        Args:
            proposed_state (np.ndarray): The proposed state.
            current_state (np.ndarray): The current state.
            symmetric (bool): Whether the proposal distribution is symmetric. Default is False.
            
        Returns:
            float: The log acceptance probability."""
        if self.symmetric:
            return np.log(self.target_dist(proposed_state)) - np.log(self.target_dist(current_state))
        else:
            log_target_term = np.log(self.target_dist(proposed_state)) - np.log(self.target_dist(current_state))
            log_proposal_term = (np.log(normal.pdf(current_state, mean=proposed_state, 
                                                   cov=np.eye(self.dim) * (self.var))) 
                                - np.log(normal.pdf(proposed_state, mean=current_state, 
                                                    cov=np.eye(self.dim) * (self.var))))

            return log_target_term + log_proposal_term
