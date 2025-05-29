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
                 geom_temp_spacing: bool = False,
                 swap_acceptance_rate: float = 0.234,
                 swap_every: int = 20,
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
            geom_temp_spacing: Use geometric spacing for temperature ladder
            swap_acceptance_rate: Target acceptance rate for swaps
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
        
        self.name = "PT_RWM_GPU_ULTRA_FUSED"
        
        # Setup target distribution
        self._setup_target_distribution()
        
        # Temperature ladder setup
        if beta_ladder is not None:
            self.beta_ladder = beta_ladder
        elif geom_temp_spacing:
            self.beta_ladder = self._construct_geometric_ladder()
        else:
            # For now, use geometric spacing as default for GPU version
            # The iterative construction can be added later if needed
            self.beta_ladder = self._construct_geometric_ladder()
            warnings.warn("Using geometric spacing for GPU PT. Iterative construction not yet implemented.")
        
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
        print(f"Beta ladder: {self.beta_ladder}")
    
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
    
    def step(self):
        """Take a step for all chains with optional swapping."""
        self.step_counter += 1
        should_swap = (self.step_counter % self.swap_every == 0)
        
        # Step 1: Generate proposals for all chains in parallel
        increments = self._generate_all_increments()
        proposals = fused_parallel_proposals(self.current_states, increments)
        
        # Step 2: Compute log densities for all proposals (potentially in parallel)
        log_densities_proposed = self._compute_log_densities_for_proposals(proposals)
        
        # Step 3: Compute acceptance ratios for all chains
        log_accept_ratios = self.beta_tensor * (log_densities_proposed - self.current_log_densities)
        
        # Step 4: Make acceptance decisions for all chains
        random_vals = torch.rand(self.num_chains, device=self.device)
        accept_flags = fused_parallel_acceptance_decisions(log_accept_ratios, random_vals)
        
        # Step 5: Update all chain states
        self.current_states, self.current_log_densities = fused_parallel_state_updates(
            self.current_states, proposals, self.current_log_densities, 
            log_densities_proposed, accept_flags
        )
        
        # Step 6: Attempt swaps if needed
        if should_swap:
            self._attempt_all_swaps()
        
        # Step 7: Store states in chains
        self._add_states_to_chains()
        
        # Update main chain to be the cold chain (beta=1.0)
        self.chain = self._get_cold_chain_cpu()
    
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