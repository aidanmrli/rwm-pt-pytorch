import numpy as np
from scipy.stats import multivariate_normal as normal
from interfaces import MHAlgorithm, TargetDistribution
from algorithms import RandomWalkMH


class ParallelTemperingRWM(MHAlgorithm):
    """Implementation of the Random Walk Metropolis-Hastings with Parallel Tempering 
    algorithm for sampling from a target distribution."""
    def __init__(
            self, 
            dim, 
            var, 
            target_dist: TargetDistribution = None, 
            symmetric=True,
            beta_ladder=None,
            geom_temp_spacing=False,
            swap_acceptance_rate=0.234,):
        super().__init__(dim, var, target_dist, symmetric)
        self.name = "PTrwm"
        ### counting variables
        self.num_swap_attempts = 0
        self.num_acceptances = 0    # use this to calculate acceptance rate
        self.acceptance_rate = 0    # this refers to SWAP acceptance rate
        self.step_counter = 0   # this is the number of steps taken by the algorithm
        self.swap_every = 20    # swap every 20 steps
        self.squared_jump_distances = 0
        self.pt_esjd = 0

        self.chains = []    # each chain is a separate RandomWalkMH instance
        self.beta_ladder = beta_ladder
        self.ideal_swap_acceptance_rate = swap_acceptance_rate  # only used for constructing the temperature ladder

        if self.beta_ladder is not None:
            for i in range(len(self.beta_ladder)):  # only if the temp ladder is not none
                self.chains.append(RandomWalkMH(dim, var, target_dist, symmetric, beta=self.beta_ladder[i]))

        else:
            self.beta_ladder = []
            if geom_temp_spacing:
                ### construct a geometrically spaced inverse temperature ladder
                beta_0, beta_min = 1, 1e-2
                curr_beta = beta_0
                c = 0.5     # set the geometric spacing constant
                while curr_beta > beta_min:
                    self.beta_ladder.append(curr_beta)
                    # initialize one RWM algorithm for each beta
                    self.chains.append(RandomWalkMH(dim, var, target_dist, symmetric, beta=self.beta_ladder[-1]))
                    curr_beta = curr_beta * c
                
                self.beta_ladder.append(beta_min)
                self.chains.append(self.chain.copy())

            else:
                ### iteratively construct the inverse temperatures
                self.construct_beta_ladder_iteratively()
                    
        self.chain = self.chains[0].chain  # the first chain is the "cold" chain
        self.log_target_density_curr_state = np.ones(len(self.beta_ladder)) * -np.inf    # store the previously computed target densities

    def get_name(self):
        """
        Return the name of the MHAlgorithm as a string.
        """
        return self.name
    
    def construct_beta_ladder_iteratively(self):
        """Construct the inverse temperature ladder iteratively 
        using a simulation-based approach."""
        beta_0, beta_min = 1, 1e-2
        curr_beta = beta_0

        while curr_beta > beta_min:
            self.beta_ladder.append(curr_beta)
            self.chains.append(RandomWalkMH(self.dim, self.var, self.target_dist, self.symmetric, beta=self.beta_ladder[-1]))  # initialize one chain for each temperature

            ### Find the next inverse temperature beta
            rho_n = 0.5
            new_beta = curr_beta / (1 + np.exp(rho_n))
            num_iters = 0

            while new_beta > beta_min:
                num_iters += 1
                curr_beta_chain = self.chain.copy()  # copy an empty chain
                new_beta_chain = self.chain.copy()  # copy an empty chain
                chains = [(curr_beta_chain, curr_beta), (new_beta_chain, new_beta)]

                ## get an average swap probability between the two chains by drawing 
                ## total_iterations samples and calculating the swap probability 
                ## for each sample
                burn_in = 0
                total_iterations = 1000    # set this to a large number.
                total_samples = total_iterations - burn_in
                for chain, beta in chains:
                    for _ in range(total_iterations):
                        # logic to draw sample is in the target distribution
                        sample = self.target_dist.draw_sample(beta)     
                        chain.append(sample)
                
                ## then calculate the swap probability between these chains for each sample
                ## and average them to get an average swap probability
                avg_swap_prob = np.zeros(total_samples)
                for i in range(total_samples):
                    ## stabilizing const is added to avoid log(0)
                    log_swap_prob = (
                        curr_beta * np.log(self.target_density(new_beta_chain[burn_in + 1 + i]) + 1e-300) + 
                        new_beta * np.log(self.target_density(curr_beta_chain[burn_in + 1 + i]) + 1e-300) -
                        curr_beta * np.log(self.target_density(curr_beta_chain[burn_in + 1 + i]) + 1e-300) -
                        new_beta * np.log(self.target_density(new_beta_chain[burn_in + 1 + i]) + 1e-300)
                        )
                    swap_prob = min(1, np.exp(log_swap_prob))
                    avg_swap_prob[i] = swap_prob

                avg_swap_prob = np.mean(avg_swap_prob)

                ### if the average swap probability is close to 0.234
                error = 0.005
                if abs(self.ideal_swap_acceptance_rate - avg_swap_prob) <= error: 
                    curr_beta = new_beta
                    print("new beta added to inverse temperature ladder: ", curr_beta, 
                          "\nSwap probability: ", avg_swap_prob,
                          "\nIdeal swap acceptance rate: ", self.ideal_swap_acceptance_rate)
                    break
                else:   ### use the recurrence defined in the paper
                    rho_n = rho_n + (avg_swap_prob - self.ideal_swap_acceptance_rate) / (num_iters ** 0.25)
                    new_beta = curr_beta / (1 + np.exp(rho_n))

            if curr_beta <= beta_min or new_beta <= beta_min:
                break

        self.beta_ladder.append(beta_min)
        self.chains.append(RandomWalkMH(self.dim, self.var, self.target_dist, self.symmetric, beta=self.beta_ladder[-1]))
        print("Finished constructing the temperature ladder.")
        print("Inverse temperature ladder: ", self.beta_ladder)


    def attempt_swap(self, j, k):
        """Attempt to swap states between two chains based on Metropolis criteria."""
        swap_prob = min(1, np.exp(self.log_swap_prob(j, k)))
        self.num_swap_attempts += 1
        if np.random.random() < swap_prob:
            # swap the states
            temp = self.chains[k].get_curr_state().copy()
            self.chains[k].set_curr_state(self.chains[j].get_curr_state().copy())
            self.chains[j].set_curr_state(temp)

            # swap the log target densities of the current state
            temp = self.chains[k].log_target_density_curr_state.copy()
            self.chains[k].log_target_density_curr_state = self.chains[j].log_target_density_curr_state.copy()
            self.chains[j].log_target_density_curr_state = temp

            self.num_acceptances += 1   # increment the number of SWAP acceptances
            self.acceptance_rate = self.num_acceptances / self.num_swap_attempts
            self.squared_jump_distances += (self.beta_ladder[j] - self.beta_ladder[k]) ** 2
            self.pt_esjd = self.squared_jump_distances / self.num_swap_attempts


    def log_swap_prob(self, j, k):
        """Calculate the log probability of swapping states between chain j and chain k."""
        log_prob = (
            self.beta_ladder[j] * self.chains[k].log_target_density_curr_state + 
            self.beta_ladder[k] * self.chains[j].log_target_density_curr_state -
            self.beta_ladder[j] * self.chains[j].log_target_density_curr_state -
            self.beta_ladder[k] * self.chains[k].log_target_density_curr_state
                )
        return log_prob
    

    def step(self):
        """Take a step for each chain. Swap states between chains every swap_every steps.
        """
        self.step_counter += 1
        swap = (self.step_counter % self.swap_every == 0)   # swap is a Boolean value

        for i in range(len(self.chains)):
            curr_chain = self.chains[i]
            # swap = False
            if swap and i < len(self.chains) - 1:
                self.attempt_swap(i, i+1)   # this handles swap acceptance rate as well
            else:
                curr_chain.step()   # handled in the RandomWalkMH instance
        
        assert self.chains[0].beta == 1, "The first chain should have beta = 1."
        self.chain = self.chains[0].chain
       