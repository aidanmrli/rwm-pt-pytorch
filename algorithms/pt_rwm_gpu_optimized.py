import torch
import torch.nn as nn
import numpy as np
import time
import warnings
from interfaces import MHAlgorithm, TargetDistribution, TorchTargetDistribution
from algorithms.rwm_gpu_optimized import RandomWalkMH_GPU_Optimized

@torch.jit.script
def fused_parallel_proposals(current_states: torch.Tensor,
                           increments: torch.Tensor) -> torch.Tensor:
    """Generate proposals for all chains in parallel."""
    return current_states + increments

@torch.jit.script
def fused_parallel_acceptance_decisions(log_accept_ratios: torch.Tensor,
                                      random_vals: torch.Tensor) -> torch.Tensor:
    """Make acceptance decisions for all chains in parallel."""
    return (log_accept_ratios > 0.0) | (random_vals < torch.exp(log_accept_ratios))

@torch.jit.script  
def fused_parallel_state_updates(current_states: torch.Tensor,
                                proposals: torch.Tensor,
                                log_densities_current: torch.Tensor,
                                log_densities_proposed: torch.Tensor,
                                accept_flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Update states for all chains based on acceptance decisions."""
    # Expand accept_flags to match state dimensions
    accept_expanded = accept_flags.unsqueeze(-1)  # Shape: (num_chains, 1)
    
    new_states = torch.where(accept_expanded, proposals, current_states)
    new_log_densities = torch.where(accept_flags, log_densities_proposed, log_densities_current)
    
    return new_states, new_log_densities

@torch.jit.script
def fused_swap_probability_calculation(log_densities: torch.Tensor,
                                     beta_ladder: torch.Tensor,
                                     chain_j: int,
                                     chain_k: int) -> torch.Tensor:
    """Calculate log swap probability between chains j and k."""
    log_prob = (
        beta_ladder[chain_j] * log_densities[chain_k] + 
        beta_ladder[chain_k] * log_densities[chain_j] -
        beta_ladder[chain_j] * log_densities[chain_j] -
        beta_ladder[chain_k] * log_densities[chain_k]
    )
    return log_prob

@torch.jit.script
def fused_swap_execution(current_states: torch.Tensor,
                        log_densities: torch.Tensor,
                        chain_j: int,
                        chain_k: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Execute swap between chains j and k."""
    # Swap states
    temp_state = current_states[chain_k].clone()
    current_states[chain_k] = current_states[chain_j]
    current_states[chain_j] = temp_state
    
    # Swap log densities
    temp_log_density = log_densities[chain_k].clone()
    log_densities[chain_k] = log_densities[chain_j]
    log_densities[chain_j] = temp_log_density
    
    return current_states, log_densities

@torch.jit.script
def ultra_fused_parallel_mcmc_step(current_states: torch.Tensor,
                                 current_log_densities: torch.Tensor,
                                 proposals: torch.Tensor,
                                 log_densities_proposed: torch.Tensor,
                                 random_vals: torch.Tensor,
                                 beta_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ultra-fused parallel MCMC step for all chains.
    
    This fuses acceptance ratio computation, acceptance decisions, and state updates
    for all parallel chains into a single kernel.
    """
    # Compute acceptance ratios for all chains
    log_accept_ratios = beta_tensor * (log_densities_proposed - current_log_densities)
    
    # Make acceptance decisions for all chains
    accept_flags = (log_accept_ratios > 0.0) | (random_vals < torch.exp(log_accept_ratios))
    
    # Update states for all chains
    accept_expanded = accept_flags.unsqueeze(-1)  # Shape: (num_chains, 1)
    new_states = torch.where(accept_expanded, proposals, current_states)
    new_log_densities = torch.where(accept_flags, log_densities_proposed, current_log_densities)
    
    return new_states, new_log_densities, accept_flags

class ParallelTemperingRWM_GPU_Optimized(MHAlgorithm):
    """
    Ultra-optimized GPU-accelerated Parallel Tempering Random Walk Metropolis.
    
    Key optimizations:
    - All chains run in parallel on GPU (true parallelization between chains)
    - JIT-compiled kernels for all operations
    - Fused operations to minimize kernel launch overhead
    - Pre-allocated GPU memory for all chains
    - Efficient swap operations using GPU tensors
    - Mixed precision support
    """
    
    def __init__(self, dim: int, var: float,
                 target_dist: TorchTargetDistribution | TargetDistribution = None,
                 symmetric: bool = True,
                 beta_ladder: list = None,
                 iterative_temp_spacing: bool = False,
                 geom_temp_spacing: bool = False,
                 swap_acceptance_rate: float = 0.234,
                 beta_min_iterative: float = 0.01,
                 N_samples_swap_est: int = 3000,
                 iterative_tolerance: float = 0.005,
                 iterative_initial_pn: float = 0.5,
                 iterative_pn_update_power: float = -0.25,
                 iterative_max_pn_steps: int = 100,
                 iterative_pn_clamp_min: float = -10.0,
                 iterative_pn_clamp_max: float = 10.0,
                 iterative_fail_tol_factor: float = 3.0,
                 swap_every: int = 100,
                 burn_in: int = 0,
                 device: str = None,
                 pre_allocate_steps: int = None,
                 dtype: torch.dtype = torch.float32):
        """Initialize the GPU-optimized Parallel Tempering RWM algorithm.
        
        Args:
            dim: Dimension of the target distribution
            var: Proposal variance
            target_dist: Target distribution to sample from
            symmetric: Whether proposal distribution is symmetric
            beta_ladder: List of inverse temperatures (if None, will be constructed)
            iterative_temp_spacing: Use iterative spacing for temperature ladder
            geom_temp_spacing: Use geometric spacing for temperature ladder
            swap_acceptance_rate: Target acceptance rate for swaps (used in iterative construction)
            beta_min_iterative: Minimum beta value for iterative construction
            N_samples_swap_est: Number of samples for swap estimation in iterative construction
            iterative_tolerance: Tolerance for iterative construction
            iterative_initial_pn: Initial pn value for iterative construction
            iterative_pn_update_power: Power for pn updates in iterative construction
            iterative_max_pn_steps: Maximum pn adjustment steps in iterative construction
            iterative_pn_clamp_min: Minimum clamp value for pn
            iterative_pn_clamp_max: Maximum clamp value for pn
            iterative_fail_tol_factor: Tolerance factor on convergence failure
            swap_every: Attempt swaps every N steps
            burn_in: Number of initial samples to discard
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
            pre_allocate_steps: Pre-allocate memory for this many steps
            dtype: Data type for computations
        """
        super().__init__(dim, var, target_dist, symmetric)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if self.device.type == 'cuda':
            print(f"Using GPU acceleration for Parallel Tempering: {torch.cuda.get_device_name()}")
            # Enable optimizations for modern GPUs
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        else:
            print("Using CPU (consider installing CUDA for optimal performance)")
        
        self.dtype = dtype
        self.burn_in = max(0, burn_in)
        self.swap_every = swap_every
        self.ideal_swap_acceptance_rate = swap_acceptance_rate
        
        # Update algorithm name based on ladder construction method
        if iterative_temp_spacing:
            self.name = "PT_RWM_GPU_ULTRA_FUSED_ITERATIVE_LADDER"
        else:
            self.name = "PT_RWM_GPU_ULTRA_FUSED"
        
        # Setup target distribution
        self._setup_target_distribution()
        
        # Temperature ladder setup
        if beta_ladder is not None:
            self.beta_ladder = beta_ladder
        elif iterative_temp_spacing:
            if not isinstance(target_dist, TorchTargetDistribution):
                raise TypeError("ParallelTemperingRWM_GPU_Optimized with iterative ladder "
                               "construction requires a TorchTargetDistribution.")
            self.beta_ladder = self._construct_iterative_ladder(
                target_swap_acceptance_rate=swap_acceptance_rate,
                beta_min=beta_min_iterative,
                N_samples_for_swap_estimation=N_samples_swap_est,
                tolerance=iterative_tolerance,
                initial_pn=iterative_initial_pn,
                pn_update_power=iterative_pn_update_power,
                max_pn_adjustment_steps=iterative_max_pn_steps,
                pn_clamping_range=(iterative_pn_clamp_min, iterative_pn_clamp_max),
                convergence_failure_tolerance_factor=iterative_fail_tol_factor
            )
        elif geom_temp_spacing:
            self.beta_ladder = self._construct_geometric_ladder()
        else:
            # Default to geometric spacing
            self.beta_ladder = self._construct_geometric_ladder()
            warnings.warn("No specific ladder construction method chosen. Using geometric spacing as default.")
        
        self.num_chains = len(self.beta_ladder)
        self.beta_tensor = torch.tensor(self.beta_ladder, device=self.device, dtype=torch.float32)
        
        # Initialize multiple chains' state on GPU
        self._initialize_gpu_chains(pre_allocate_steps)
        
        # Swap tracking
        self.num_swap_attempts = 0
        self.num_swap_acceptances = 0
        self.swap_acceptance_rate = 0.0
        self.step_counter = 0
        self.squared_jump_distances = 0.0
        self.pt_esjd = 0.0
        
        # Precomputed randoms for swaps
        self.precomputed_swap_randoms = None
        self.swap_random_index = 0
        
        print(f"Initialized PT-RWM with {self.num_chains} chains")
        print(f"Beta ladder: [{', '.join(f'{b:.6f}' for b in self.beta_ladder)}]")
    
    def _setup_target_distribution(self):
        """Setup target distribution evaluation method."""
        if isinstance(self.target_dist, TorchTargetDistribution):
            self.use_torch_target = True
            self.target_dist.to(self.device)
        else:
            self.use_torch_target = False
            warnings.warn("Using legacy CPU target distribution - GPU acceleration limited")
    
    def _construct_geometric_ladder(self):
        """Construct geometrically spaced inverse temperature ladder."""
        beta_0, beta_min = 1.0, 1e-2
        curr_beta = beta_0
        c = 0.5  # Geometric spacing constant
        ladder = []
        
        while curr_beta > beta_min:
            ladder.append(curr_beta)
            curr_beta = curr_beta * c
        
        ladder.append(beta_min)
        return ladder
    
    def _get_typical_samples_at_beta(self, beta_val: float, N_samples: int) -> torch.Tensor:
        """
        Helper function to get 'typical' samples from pi(x) effectively tempered by beta_val,
        by using the target distribution's specialized sampler.
        
        Args:
            beta_val: The inverse temperature.
            N_samples: The number of typical samples to return.

        Returns:
            torch.Tensor: Samples of shape (N_samples, self.dim) on self.device.
        """
        if not hasattr(self.target_dist, 'draw_samples_torch'):
            # This check is also in _construct_iterative_ladder, but good for direct calls too
            raise NotImplementedError(
                "The target distribution must implement 'draw_samples_torch(n_samples, beta)' "
                "for iterative temperature ladder construction."
            )
        
        # draw_samples_torch is expected to handle the tempering by beta_val internally
        # (e.g., by scaling component variances by 1/beta_val)
        samples = self.target_dist.draw_samples_torch(N_samples, beta=beta_val)
        return samples.to(self.device)  # Ensure samples are on the correct device
    
    def _construct_iterative_ladder(self,
                                   target_swap_acceptance_rate: float,
                                   beta_min: float,
                                   N_samples_for_swap_estimation: int,
                                   tolerance: float,
                                   initial_pn: float,
                                   pn_update_power: float,
                                   max_pn_adjustment_steps: int,
                                   pn_clamping_range: tuple,
                                   convergence_failure_tolerance_factor: float
                                   ) -> list:
        """
        Constructs an inverse temperature ladder (beta_ladder) iteratively.
        Aims for a target_swap_acceptance_rate between adjacent temperatures.
        """
        
        print(f"\n[Ladder Construction] Starting iterative beta ladder construction...")
        print(f"  Target Swap AR: {target_swap_acceptance_rate:.4f}, Beta Min: {beta_min:.4f}, N_samples_est: {N_samples_for_swap_estimation}")
        print(f"  Tolerance: {tolerance:.4f}, P_n_init: {initial_pn:.2f}, P_n_power: {pn_update_power:.2f}")
        print(f"  Max P_n Steps: {max_pn_adjustment_steps}, P_n Clamp: {pn_clamping_range}")

        beta_ladder = [1.0]
        beta_curr = 1.0
        
        outer_loop_iter = 0
        while True:  # Outer loop for adding betas to the ladder
            outer_loop_iter += 1
            print(f"\n[Ladder Outer Loop {outer_loop_iter}] Current beta_curr = {beta_curr:.6f} (Ladder length: {len(beta_ladder)})")
            
            if beta_curr <= beta_min + 1e-6:  # Check if beta_curr itself is already at or below beta_min
                print(f"  beta_curr ({beta_curr:.6f}) is already at or below beta_min ({beta_min:.4f}). Stopping outer loop.")
                break

            pn = initial_pn 
            n_pn_updates = 1 
            found_next_beta_successfully = False
            
            # Variables to store results from the last iteration of the inner loop
            last_beta_star_candidate = -1.0 
            last_avg_swap_prob_candidate = -1.0

            for adjustment_iter in range(1, max_pn_adjustment_steps + 1):  # Inner loop for p_n adjustment
                clamped_pn = np.clip(pn, pn_clamping_range[0], pn_clamping_range[1]) 
                
                if beta_curr < 1e-9: 
                    print(f"  [P_n Loop {adjustment_iter}] Error: beta_curr ({beta_curr:.2e}) is too small. Aborting ladder construction.")
                    last_beta_star_candidate = -1.0  # Error flag
                    break 

                denominator = 1.0 + np.exp(clamped_pn)
                if denominator < 1e-9: 
                    print(f"  [P_n Loop {adjustment_iter}] Warning: Denominator (1+exp(pn={clamped_pn:.4f})) is near zero ({denominator:.2e}).")
                    # Heuristic: if pn caused this, it's likely too large positive, making exp(pn) huge,
                    # leading to beta_star being too small. Try to reduce pn.
                    # Or if clamped_pn was very negative, exp(pn) ~0, denom ~1. This case is rare for denom near zero.
                    pn = clamped_pn - 0.5 * abs(clamped_pn) if clamped_pn != 0 else -0.5  # More aggressive reduction
                    print(f"     Adjusted pn to: {pn:.4f}. Retrying denominator calculation.")
                    clamped_pn = np.clip(pn, pn_clamping_range[0], pn_clamping_range[1])
                    denominator = 1.0 + np.exp(clamped_pn)
                    if denominator < 1e-9:
                        print(f"  [P_n Loop {adjustment_iter}] Error: Denominator still too small after adjustment. Aborting for this beta_curr.")
                        last_beta_star_candidate = -1.0  # Error flag
                        break 
                
                beta_star = beta_curr / denominator
                last_beta_star_candidate = beta_star  # Store for potential use if max_pn_steps is hit

                print(f"  [P_n Loop {adjustment_iter}/{max_pn_adjustment_steps}] pn={pn:.6f} (clamped: {clamped_pn:.6f}), Proposed beta* = {beta_star:.6f}")

                if beta_star < beta_min:
                    print(f"    Proposed beta* ({beta_star:.6f}) < beta_min ({beta_min:.4f}). Will not add. Inner loop terminates.")
                    break  # Break from inner p_n adjustment loop, last_beta_star_candidate will be < beta_min

                # Estimate Swap Acceptance Probability 'a'
                self.target_dist.to(self.device)  # Ensure target is on device for sampling & density

                samples_star = self._get_typical_samples_at_beta(beta_star, N_samples_for_swap_estimation)
                samples_curr = self._get_typical_samples_at_beta(beta_curr, N_samples_for_swap_estimation)

                log_pi_at_samples_star = self.target_dist.log_density(samples_star)
                log_pi_at_samples_curr = self.target_dist.log_density(samples_curr)
                
                log_r_k = (beta_curr - beta_star) * (log_pi_at_samples_star - log_pi_at_samples_curr)
                
                swap_probs = torch.exp(torch.clamp_max(log_r_k, 0.0))  # min(1, exp(log_r_k))
                current_avg_swap_prob = torch.mean(swap_probs).item()
                last_avg_swap_prob_candidate = current_avg_swap_prob
                
                print(f"    Estimated avg swap prob 'a' = {current_avg_swap_prob:.4f} (Target: {target_swap_acceptance_rate:.4f})")

                if abs(current_avg_swap_prob - target_swap_acceptance_rate) <= tolerance:
                    print(f"    Swap prob {current_avg_swap_prob:.4f} is within tolerance {tolerance:.4f}. Adding beta* = {beta_star:.6f}")
                    beta_ladder.append(beta_star)
                    beta_curr = beta_star
                    found_next_beta_successfully = True
                    break 
                else:
                    adjustment = (n_pn_updates ** pn_update_power) * (current_avg_swap_prob - target_swap_acceptance_rate)
                    pn = pn + adjustment
                    n_pn_updates += 1
            # End of Inner Loop (p_n adjustment)

            if not found_next_beta_successfully:
                # Inner loop finished without meeting strict tolerance
                if adjustment_iter == max_pn_adjustment_steps and \
                   last_beta_star_candidate >= beta_min and \
                   last_beta_star_candidate != -1.0:  # -1.0 is our error flag
                    
                    print(f"  Max p_n adjustment steps ({max_pn_adjustment_steps}) reached for current beta_curr.")
                    print(f"    Last candidate beta* = {last_beta_star_candidate:.6f}, "
                          f"with swap_prob = {last_avg_swap_prob_candidate:.4f}")
                    
                    if abs(last_avg_swap_prob_candidate - target_swap_acceptance_rate) <= tolerance * convergence_failure_tolerance_factor:
                        print(f"    Accepting candidate beta* as it's within wider tolerance ({tolerance * convergence_failure_tolerance_factor:.4f}).")
                        beta_ladder.append(last_beta_star_candidate)
                        beta_curr = last_beta_star_candidate
                        # Continue to the next iteration of the outer loop
                    else:
                        print(f"    Candidate beta* not within wider tolerance. Stopping ladder construction.")
                        break  # Break outer loop
                else:
                    # Inner loop broke due to beta_star < beta_min, or a fatal error (-1.0)
                    print(f"  Inner loop terminated. Reason: Proposed beta* ({last_beta_star_candidate:.6f}) "
                          f"vs beta_min ({beta_min:.4f}) or fatal error. Stopping ladder construction.")
                    break  # Break outer loop
        # End of Outer Loop

        # Finalization
        if not beta_ladder:  # Should ideally not be empty if we start with 1.0
            print("[Ladder Construction] Error: Beta ladder is empty after construction attempt. Returning default.")
            return [1.0, beta_min] 

        # Add beta_min if it's not already the last element (or very close) AND the current last beta is greater
        if beta_ladder[-1] > beta_min + 1e-5:  # Check if last beta is meaningfully greater than beta_min
            print(f"[Ladder Construction] Finalizing ladder by appending beta_min = {beta_min:.6f} (last current beta is {beta_ladder[-1]:.6f})")
            beta_ladder.append(beta_min)
        elif beta_ladder[-1] < beta_min - 1e-5:  # Should not happen if logic is correct and beta_star checks work
             warnings.warn(f"[Ladder Construction] Warning: Last beta in ladder ({beta_ladder[-1]:.6f}) is already less than beta_min ({beta_min:.6f}). Not appending beta_min.")
        else:
             print(f"[Ladder Construction] Last beta in ladder ({beta_ladder[-1]:.6f}) is already close to or at beta_min ({beta_min:.4f}). No final append needed.")

        print(f"\n[Ladder Construction] Finished. Constructed beta_ladder (length {len(beta_ladder)}):")
        print(f"  [{', '.join(f'{b:.6f}' for b in beta_ladder)}]")
        return beta_ladder
    
    def _initialize_gpu_chains(self, pre_allocate_steps):
        """Initialize all chains' state and memory on GPU."""
        # Current states for all chains
        self.current_states = torch.zeros(
            (self.num_chains, self.dim), 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Current log densities for all chains
        self.current_log_densities = torch.full(
            (self.num_chains,), 
            -float('inf'), 
            device=self.device, 
            dtype=torch.float32
        )
        
        # Proposal covariance matrices for all chains (scaled by beta)
        self.proposal_covs_chol = torch.zeros(
            (self.num_chains, self.dim, self.dim),
            device=self.device,
            dtype=self.dtype
        )
        
        # Setup covariance for each chain
        for i, beta in enumerate(self.beta_ladder):
            cov = (self.var / beta) * torch.eye(self.dim, device=self.device, dtype=torch.float32)
            self.proposal_covs_chol[i] = torch.linalg.cholesky(cov).to(self.dtype)
        
        # Pre-allocate chain storage if requested
        self.pre_allocate_steps = pre_allocate_steps
        if pre_allocate_steps:
            total_allocation = self.burn_in + pre_allocate_steps + 1
            self.pre_allocated_chains = torch.zeros(
                (self.num_chains, total_allocation, self.dim),
                device=self.device,
                dtype=self.dtype
            )
            self.pre_allocated_log_densities = torch.zeros(
                (self.num_chains, total_allocation),
                device=self.device,
                dtype=torch.float32
            )
            self.chain_indices = torch.zeros(self.num_chains, dtype=torch.long)
        else:
            self.pre_allocated_chains = None
            self.pre_allocated_log_densities = None
            self.chain_indices = None
        
        # Initialize with first sample from base class
        initial_state = torch.tensor(
            self.chain[-1], device=self.device, dtype=self.dtype
        )
        self.current_states[:] = initial_state  # All chains start from same point
        
        # Compute initial log densities for all chains
        self._compute_all_log_densities()
        
        # Add initial states to pre-allocated storage
        if self.pre_allocated_chains is not None:
            self.pre_allocated_chains[:, 0] = self.current_states
            self.pre_allocated_log_densities[:, 0] = self.current_log_densities
            self.chain_indices[:] = 1
    
    def _compute_all_log_densities(self):
        """Compute log densities for all current states in parallel."""
        if self.use_torch_target:
            # Batch computation for all chains
            self.current_log_densities = self.target_dist.log_density(self.current_states)
        else:
            # Fallback to sequential computation for legacy distributions
            for i in range(self.num_chains):
                state_numpy = self.current_states[i].detach().cpu().numpy()
                density = self.target_density(state_numpy)
                self.current_log_densities[i] = torch.log(
                    torch.tensor(density + 1e-300, device=self.device, dtype=torch.float32)
                )
    
    def _compute_log_densities_for_proposals(self, proposals):
        """Compute log densities for proposal states in parallel."""
        if self.use_torch_target:
            # Batch computation for all proposals
            return self.target_dist.log_density(proposals)
        else:
            # Fallback to sequential computation
            log_densities = torch.zeros(self.num_chains, device=self.device, dtype=torch.float32)
            for i in range(self.num_chains):
                state_numpy = proposals[i].detach().cpu().numpy()
                density = self.target_density(state_numpy)
                log_densities[i] = torch.log(
                    torch.tensor(density + 1e-300, device=self.device, dtype=torch.float32)
                )
            return log_densities
    
    def get_name(self):
        return self.name
    
    def reset(self):
        """Reset the algorithm to initial state."""
        super().reset()
        self.num_swap_attempts = 0
        self.num_swap_acceptances = 0
        self.swap_acceptance_rate = 0.0
        self.step_counter = 0
        self.squared_jump_distances = 0.0
        self.pt_esjd = 0.0
        if self.pre_allocated_chains is not None:
            self.chain_indices[:] = 0
        self.precomputed_swap_randoms = None
        self.swap_random_index = 0
        # Clear chain cache for performance fix
        self._chain_cache = None
    
    def step(self):
        """Take a step for all chains with optional swapping."""
        self.step_counter += 1
        should_swap = (self.step_counter % self.swap_every == 0)
        
        # Step 1: Generate proposals for all chains in parallel
        increments = self._generate_all_increments()
        proposals = self.current_states + increments
        
        # Step 2: Compute log densities for all proposals (potentially in parallel)
        log_densities_proposed = self._compute_log_densities_for_proposals(proposals)
        
        # Step 3-5 FUSED: Compute acceptance ratios, make decisions, and update states in single kernel
        random_vals = torch.rand(self.num_chains, device=self.device)
        self.current_states, self.current_log_densities, accept_flags = ultra_fused_parallel_mcmc_step(
            self.current_states,
            self.current_log_densities,
            proposals,
            log_densities_proposed,
            random_vals,
            self.beta_tensor
        )
        
        # Step 6: Attempt swaps if needed
        if should_swap:
            self._attempt_all_swaps()
        
        # Step 7: Store states in chains
        self._add_states_to_chains()
    
    def _generate_all_increments(self):
        """Generate proposal increments for all chains in parallel."""
        # Generate random vectors for all chains
        raw_increments = torch.randn(
            self.num_chains, self.dim, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Apply covariance structure for each chain
        increments = torch.zeros_like(raw_increments)
        for i in range(self.num_chains):
            increments[i] = torch.matmul(self.proposal_covs_chol[i], raw_increments[i])
        
        return increments
    
    def _attempt_all_swaps(self):
        """Attempt swaps between adjacent chains."""
        # Get precomputed random value for swap decision
        swap_random = self._get_next_swap_random()
        
        # Attempt swaps between adjacent chains
        for j in range(self.num_chains - 1):
            k = j + 1
            
            # Calculate swap probability
            log_swap_prob = fused_swap_probability_calculation(
                self.current_log_densities, self.beta_tensor, j, k
            )
            
            swap_prob = torch.min(torch.ones(1, device=self.device), torch.exp(log_swap_prob))
            
            self.num_swap_attempts += 1
            
            if swap_random < swap_prob:
                # Execute swap
                self.current_states, self.current_log_densities = fused_swap_execution(
                    self.current_states, self.current_log_densities, j, k
                )
                
                self.num_swap_acceptances += 1
                self.swap_acceptance_rate = self.num_swap_acceptances / self.num_swap_attempts
                
                # Update ESJD tracking
                beta_diff = self.beta_ladder[j] - self.beta_ladder[k]
                self.squared_jump_distances += beta_diff ** 2
                self.pt_esjd = self.squared_jump_distances / self.num_swap_attempts
            
            # Update random for next swap attempt
            swap_random = self._get_next_swap_random()
    
    def _get_next_swap_random(self):
        """Get next precomputed random value for swap decisions."""
        if (self.precomputed_swap_randoms is not None and 
            self.swap_random_index < self.precomputed_swap_randoms.shape[0]):
            random_val = self.precomputed_swap_randoms[self.swap_random_index]
            self.swap_random_index += 1
            return random_val
        else:
            return torch.rand(1, device=self.device)
    
    def _add_states_to_chains(self):
        """Add current states to chain storage."""
        if self.pre_allocated_chains is not None:
            # Add to pre-allocated storage
            for i in range(self.num_chains):
                if self.chain_indices[i] < self.pre_allocated_chains.shape[1]:
                    self.pre_allocated_chains[i, self.chain_indices[i]] = self.current_states[i]
                    self.pre_allocated_log_densities[i, self.chain_indices[i]] = self.current_log_densities[i]
                    self.chain_indices[i] += 1
        else:
            # Dynamic allocation (less efficient)
            if not hasattr(self, 'gpu_chains'):
                self.gpu_chains = [[] for _ in range(self.num_chains)]
            
            for i in range(self.num_chains):
                self.gpu_chains[i].append(self.current_states[i].clone())
        
        # Invalidate chain cache since we added new states
        self._chain_cache = None
    
    def _get_cold_chain_cpu(self):
        """Get the cold chain (beta=1.0) as CPU numpy array for compatibility."""
        if self.pre_allocated_chains is not None:
            cold_chain = self.pre_allocated_chains[0, :self.chain_indices[0]]
            return cold_chain.detach().cpu().numpy().tolist()
        else:
            if hasattr(self, 'gpu_chains'):
                return [state.detach().cpu().numpy() for state in self.gpu_chains[0]]
            else:
                return []
    
    def get_all_chains_gpu(self):
        """Get all chains as GPU tensors."""
        if self.pre_allocated_chains is not None:
            return [self.pre_allocated_chains[i, :self.chain_indices[i]] 
                   for i in range(self.num_chains)]
        else:
            if hasattr(self, 'gpu_chains'):
                return [torch.stack(chain) for chain in self.gpu_chains]
            else:
                return []
    
    def get_cold_chain_gpu(self):
        """Get the cold chain (beta=1.0) as GPU tensor."""
        chains = self.get_all_chains_gpu()
        return chains[0] if chains else torch.empty(0, self.dim, device=self.device)
    
    @property
    def chain(self):
        """Lazy property for chain access - only transfers from GPU when needed."""
        if not hasattr(self, '_chain_cache') or self._chain_cache is None:
            self._chain_cache = self._get_cold_chain_cpu()
        return self._chain_cache
    
    @chain.setter  
    def chain(self, value):
        """Allow setting the chain directly."""
        self._chain_cache = value
    
    def generate_samples(self, num_samples: int):
        """Generate samples using ultra-optimized parallel tempering.
        
        Args:
            num_samples: Total number of samples to generate (AFTER burn-in)
            
        Returns:
            Chain of samples from the cold chain (EXCLUDING burn-in samples)
        """
        print(f"Generating {num_samples} samples (+ {self.burn_in} burn-in) using GPU Parallel Tempering")
        print(f"Running {self.num_chains} chains in parallel with swaps every {self.swap_every} steps")
        
        # Pre-compute swap random numbers
        total_steps = self.burn_in + num_samples
        max_swaps = total_steps // self.swap_every * (self.num_chains - 1)
        self.precomputed_swap_randoms = torch.rand(max_swaps + 100, device=self.device)  # +100 for safety
        self.swap_random_index = 0
        
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
        
        start_time = time.time()
        
        # Run parallel tempering
        for i in range(total_steps):
            self.step()
            
            if (i + 1) % 1000 == 0:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed
                if self.step_counter > self.burn_in:
                    print(f"Generated {i + 1}/{total_steps} samples "
                          f"(Rate: {rate:.1f} samples/sec, Swap Accept: {self.swap_acceptance_rate:.3f})")
                else:
                    print(f"Burn-in: {i + 1}/{self.burn_in} samples "
                          f"(Rate: {rate:.1f} samples/sec)")
        
        if self.device.type == 'cuda':
            end_event.record()
            torch.cuda.synchronize()
            gpu_time = start_event.elapsed_time(end_event) / 1000.0
            print(f"GPU kernel time: {gpu_time:.3f}s ({total_steps/gpu_time:.1f} samples/sec)")
        
        # Return only post-burn-in samples from cold chain
        cold_chain = self.get_cold_chain_gpu()
        
        # Calculate offset: skip initial state + burn-in samples
        burn_in_offset = 1 + self.burn_in
        
        # PERFORMANCE FIX: Set the cold chain for parent class compatibility ONLY at the end
        # This avoids the expensive CPU transfer that was happening every step
        self.chain = self._get_cold_chain_cpu()
        
        return cold_chain[burn_in_offset:]
    
    def expected_squared_jump_distance_gpu(self):
        """Compute ESJD for the cold chain using GPU operations."""
        cold_chain = self.get_cold_chain_gpu()
        
        # Remove burn-in samples
        if cold_chain.shape[0] > self.burn_in + 1:
            chain_tensor = cold_chain[self.burn_in:]
        else:
            raise ValueError(f"Insufficient post-burn-in samples")
        
        if chain_tensor.shape[0] < 2:
            return 0.0
        
        # Compute ESJD
        diff = chain_tensor[1:] - chain_tensor[:-1]
        squared_jumps = torch.sum(diff * diff, dim=1)
        
        return torch.mean(squared_jumps).item()
    
    def get_diagnostic_info(self):
        """Get detailed diagnostic information."""
        info = {
            'device': str(self.device),
            'dtype': str(self.dtype),
            'algorithm': self.name,
            'num_chains': self.num_chains,
            'beta_ladder': self.beta_ladder,
            'swap_every': self.swap_every,
            'step_counter': self.step_counter,
            'swap_acceptance_rate': self.swap_acceptance_rate,
            'pt_esjd': self.pt_esjd,
            'optimization_level': 'ULTRA_FUSED_PARALLEL_CHAINS',
            'parallel_processing': f'{self.num_chains} chains processed in parallel',
            'kernel_fusion': 'Fused operations for proposals, acceptances, and swaps',
            'memory_allocated_mb': torch.cuda.memory_allocated() / 1e6 if self.device.type == 'cuda' else 0,
        }
        return info
    
    def performance_summary(self):
        """Print performance summary."""
        print("\n" + "="*70)
        print("ULTRA-FUSED PARALLEL TEMPERING RWM - PERFORMANCE SUMMARY")
        print("="*70)
        print(f"Algorithm: {self.name}")
        print(f"Device: {self.device}")
        print(f"Chains: {self.num_chains} running in parallel")
        print(f"Temperature ladder: {[f'{b:.3f}' for b in self.beta_ladder]}")
        print("\nParallel Processing Benefits:")
        print(f"  ✓ {self.num_chains} chains computed simultaneously")
        print("  ✓ Vectorized proposal generation across all chains")
        print("  ✓ Parallel density evaluations for all chains")
        print("  ✓ Fused acceptance decisions for all chains")
        print("  ✓ Optimized swap operations on GPU")
        print("\nOptimization Techniques:")
        print("  ✓ JIT Compilation for all critical operations")
        print("  ✓ Ultra-Fused Kernels: Combined acceptance ratios, decisions, and updates")
        print("  ✓ Kernel Fusion for parallel chain updates")
        print("  ✓ Pre-allocated GPU memory for all chains")
        print("  ✓ Batch random number generation")
        print("  ✓ Efficient tensor operations for swaps")
        if self.device.type == 'cuda':
            print("  ✓ CUDA optimizations enabled")
        print(f"\nPerformance Metrics:")
        print(f"  • Swap acceptance rate: {self.swap_acceptance_rate:.3f}")
        print(f"  • PT-ESJD: {self.pt_esjd:.6f}")
        print(f"  • Steps completed: {self.step_counter}")
        print("="*70) 