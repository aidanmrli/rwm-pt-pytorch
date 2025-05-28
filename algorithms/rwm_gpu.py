import torch
import torch.nn as nn
import numpy as np
from interfaces import MHAlgorithm, TargetDistribution, TorchTargetDistribution
import warnings

class RandomWalkMH_GPU(MHAlgorithm):
    """GPU-accelerated implementation of Random Walk Metropolis-Hastings algorithm.
    
    This implementation provides GPU acceleration for true standard RWM through:
    - GPU acceleration for density evaluations
    - GPU memory pre-allocation for chains
    - GPU tensor operations for proposals and acceptances
    """
    
    def __init__(self, dim: int, var: float, 
                 target_dist: TorchTargetDistribution | TargetDistribution = None, 
                 symmetric: bool = True,
                 burn_in: int = 0, 
                 beta: float = 1.0, beta_ladder: float = None, swap_acceptance_rate: float = None,
                 device: str = None, pre_allocate_steps: int = None, standard_rwm: bool = True,
                 ):
        """Initialize the GPU-accelerated RandomWalkMH algorithm.
        
        Args:
            dim: Dimension of the target distribution
            var: Proposal variance
            target_dist: Target distribution to sample from (TargetDistribution or TorchTargetDistribution)
            symmetric: Whether proposal distribution is symmetric
            beta: Temperature parameter (inverse)
            burn_in: Number of initial samples to discard for MCMC burn-in (default: 0)
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
            pre_allocate_steps: Pre-allocate memory for this many steps (None for dynamic).
            NOTE: This should be set to the number of samples you want to generate.
            standard_rwm: If True, use standard RWM (sequential). 
            
        """
        super().__init__(dim, var, target_dist, symmetric)
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if self.device.type == 'cuda':
            print(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU (consider installing CUDA for better performance)")
        
        self.beta = beta
        self.standard_rwm = standard_rwm
        
        if standard_rwm:
            self.name = "RWM_GPU_Standard"
        else:
            raise ValueError("Batch processing is not supported for GPU-accelerated RWM")
        
        # Performance tracking
        self.num_acceptances = 0
        self.acceptance_rate = 0
        self.total_steps = 0
        self.burn_in = max(0, burn_in)
        
        # Pre-allocate memory if requested
        self.pre_allocate_steps = pre_allocate_steps
        if pre_allocate_steps:
            # Allocate memory for burn_in + pre_allocate_steps + 1 (initial state)
            total_allocation = self.burn_in + pre_allocate_steps + 1
            self.pre_allocated_chain = torch.zeros(
                (total_allocation, dim), 
                device=self.device, 
                dtype=torch.float32
            )
            self.chain_index = 0
        else:
            self.pre_allocated_chain = None
            self.chain_index = None
            
        # Convert proposal covariance to GPU
        # NOTE: To sample from a multivariate normal distribution N(μ, Σ), 
        # you can use the transformation: x = μ + L * z, 
        # where z ~ N(0, I) and L is the Cholesky decomposition of Σ.
        self.proposal_cov = (var / beta) * torch.eye(dim, device=self.device, dtype=torch.float32)
        self.proposal_cov_chol = torch.linalg.cholesky(self.proposal_cov)
        
        self.current_state = None
        self.log_target_density_current = -float('inf')
        
        # Setup target distribution evaluation method
        self._setup_target_distribution()
        
        # Increment proposal optimization: pre-computed increments
        self.precomputed_increments = None
        self.increment_index = 0
    
    def _setup_target_distribution(self):
        """Setup target distribution evaluation method based on type."""
        if isinstance(self.target_dist, TorchTargetDistribution):
            # Modern PyTorch-native target distribution
            self.use_torch_target = True
            # Ensure target distribution is on the same device
            self.target_dist.to(self.device)
        else:
            # Legacy CPU target distribution
            self.use_torch_target = False
    
    def get_name(self):
        return self.name
    
    def reset(self):
        """Reset the algorithm to initial state."""
        super().reset()
        self.num_acceptances = 0
        self.acceptance_rate = 0
        self.total_steps = 0
        self.current_state = None
        self.log_target_density_current = -float('inf')
        if self.pre_allocated_chain is not None:
            self.chain_index = 0
        # Reset increment proposal state
        self.precomputed_increments = None
        self.increment_index = 0
    
    def step(self):
        """Take a single MCMC step using the selected method."""
        if self.standard_rwm:
            self._standard_step()
        else:
            raise ValueError("Batch processing is not supported for GPU-accelerated RWM")
    
    def _standard_step(self):
        """Take a single standard RWM step with GPU acceleration."""
        # Initialize current state if needed
        if self.current_state is None:
            # Use the initial sample from the base class
            self.current_state = torch.tensor(
                self.chain[-1], device=self.device, dtype=torch.float32
            )
            self.log_target_density_current = self._compute_log_density(self.current_state)
            
            # Add initial state to pre-allocated chain if using pre-allocation
            if self.pre_allocated_chain is not None and self.chain_index == 0:
                self.pre_allocated_chain[self.chain_index] = self.current_state
                self.chain_index += 1
        
        proposal = self._generate_single_proposal()
        log_accept_ratio, log_density_proposed = self._compute_acceptance_ratio(proposal)
        accept = self._make_single_acceptance_decision(log_accept_ratio)
        self.total_steps += 1

        if accept:
            self.current_state = proposal
            self.log_target_density_current = log_density_proposed.item()
            if self.total_steps > self.burn_in:
                self.num_acceptances += 1
        
        self._add_to_chain(self.current_state)
        if self.total_steps > self.burn_in:
            post_burnin_steps = self.total_steps - self.burn_in
            if post_burnin_steps > 0:
                self.acceptance_rate = self.num_acceptances / post_burnin_steps
    
    def _generate_single_proposal(self):
        """Generate a single proposal state using GPU operations."""
        if self.precomputed_increments is not None and self.increment_index < self.precomputed_increments.shape[0]:
            # Use pre-computed increment for optimal GPU performance
            increment = self.precomputed_increments[self.increment_index]
            self.increment_index += 1
            proposal = self.current_state + increment
        else:
            # Fallback to on-demand generation (less efficient)
            noise = torch.randn(self.dim, device=self.device, dtype=torch.float32)
            proposal = self.current_state + torch.matmul(self.proposal_cov_chol, noise)
        
        return proposal
    
    def _compute_acceptance_ratio(self, proposal):
        """Compute log acceptance ratio for a single proposal."""
        # Compute log density for proposal using the appropriate method
        if self.use_torch_target:
            # Modern PyTorch-native target distribution - use log_density directly
            log_density_proposed = self.target_dist.log_density(proposal)
        else:
            # Legacy CPU target distribution
            density_proposed = self._evaluate_target_density_cpu(proposal)
            log_density_proposed = torch.log(torch.tensor(density_proposed + 1e-300, device=self.device, dtype=torch.float32))
        
        # Compute acceptance ratio
        log_accept_ratio = self.beta * (log_density_proposed - self.log_target_density_current)
        
        if not self.symmetric:
            # Add proposal ratio terms (currently assuming symmetric proposals)
            warnings.warn("Asymmetric proposals not yet optimized for GPU implementation")
            raise NotImplementedError("Asymmetric proposals not yet optimized for GPU implementation")
        
        return log_accept_ratio, log_density_proposed
    
    def _make_single_acceptance_decision(self, log_accept_ratio):
        """Make acceptance/rejection decision for a single proposal."""
        # Generate random number for acceptance decision
        random_val = torch.rand(1, device=self.device)
        
        # Accept if log_ratio > 0 or random < exp(log_ratio)
        accept = (log_accept_ratio > 0) or (random_val < torch.exp(log_accept_ratio))
        
        return accept
    
    def _make_acceptance_decisions(self, log_accept_ratios):
        """Make vectorized acceptance/rejection decisions."""
        # Generate random numbers for acceptance decisions
        random_vals = torch.rand(log_accept_ratios.shape[0], device=self.device)
        
        # Accept if log_ratio > 0 or random < exp(log_ratio)
        accept_flags = (log_accept_ratios > 0) | (random_vals < torch.exp(log_accept_ratios))
        
        return accept_flags
    
    def _evaluate_target_density_cpu(self, state):
        """Evaluate target density, handling GPU tensor to CPU conversion if needed.
        This is a legacy function that is used to evaluate the target density for CPU target distributions.
        It is not used for GPU target distributions.
        """
        if isinstance(state, torch.Tensor):
            # Convert GPU tensor to numpy for legacy CPU target distributions
            if state.dim() == 1:
                # Single state
                state_numpy = state.detach().cpu().numpy()
                return self.target_density(state_numpy)
            else:
                # Batch of states
                raise NotImplementedError("Batch processing is not supported for GPU-accelerated RWM")
                state_numpy = state.detach().cpu().numpy()
                return np.array([self.target_density(s) for s in state_numpy])
        else:
            # Already numpy
            return self.target_density(state)
    
    def _compute_log_density(self, state):
        """Compute log density for a single state."""
        if self.use_torch_target:
            # Modern PyTorch-native target distribution - use log_density directly
            log_density = self.target_dist.log_density(state)
        else:
            density = self._evaluate_target_density_cpu(state)
            log_density = torch.log(torch.tensor(density + 1e-300, device=self.device, dtype=torch.float32))
        
        return log_density.item()
    
    def _add_to_chain(self, state):
        """Add state to the chain (either pre-allocated or dynamic)."""
        if self.pre_allocated_chain is not None:
            if self.chain_index < self.pre_allocated_chain.shape[0]:
                self.pre_allocated_chain[self.chain_index] = state
                self.chain_index += 1
            else:
                warnings.warn("Pre-allocated chain full, switching to dynamic allocation")
                self.pre_allocated_chain = None
                if self.use_torch_target:
                    self.chain.append(state)
                else:
                    self.chain.append(state.detach().cpu().numpy())
        else:
            # Convert GPU tensor to CPU numpy array for compatibility with base class
            self.chain.append(state.detach().cpu().numpy())
    
    def get_chain_gpu(self):
        """Get the chain as a GPU tensor."""
        if self.pre_allocated_chain is not None:
            return self.pre_allocated_chain[:self.chain_index]
        else:
            return torch.tensor(np.array(self.chain), device=self.device, dtype=torch.float32)
    
    def generate_samples(self, num_samples):
        """Generate samples.
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            Chain of samples
        """
        if self.standard_rwm:
            return self.generate_samples_standard(num_samples)
        else:
            raise
    
    def generate_samples_standard(self, num_samples):
        """Generate samples using standard sequential RWM with GPU acceleration.
        
        Args:
            num_samples: Total number of samples to generate (AFTER burn-in)
            
        Returns:
            Chain of samples as a torch tensor (EXCLUDING burn-in samples)
        """
        print(f"Generating {num_samples} samples (+ {self.burn_in} burn-in) using standard RWM with GPU acceleration")
        
        # Pre-compute all increments for maximum GPU efficiency
        total_steps = self.burn_in + num_samples
        self._precompute_increments(total_steps)
        
        for i in range(total_steps):
            self._standard_step()
            
            if (i + 1) % 1000 == 0:
                if self.total_steps > self.burn_in:
                    print(f"Generated {i + 1}/{total_steps} samples "
                          f"(Accept rate: {self.acceptance_rate:.3f})")
                else:
                    print(f"Burn-in: {i + 1}/{self.burn_in} samples")
        
        # Return the correct chain based on allocation method
        if self.pre_allocated_chain is not None:
            # Return only post-burn-in samples
            return self.pre_allocated_chain[self.burn_in:self.chain_index]
        else:
            # Return only post-burn-in samples
            return self.chain[self.burn_in:]
    
    def expected_squared_jump_distance_gpu(self):
        """Compute ESJD using GPU tensors for efficiency, removing burn-in samples."""
        if self.pre_allocated_chain is not None and self.chain_index > 1:
            if self.chain_index > self.burn_in:
                chain_tensor = self.pre_allocated_chain[self.burn_in:self.chain_index]
            else:
                raise ValueError(f"Insufficient post-burn-in samples: chain_index={self.chain_index}, burn_in={self.burn_in}. Need at least {self.burn_in + 2} total samples.")
        else:
            chain_tensor = self.get_chain_gpu()
            if chain_tensor.shape[0] > self.burn_in + 1:  # Need at least 2 post-burn-in samples
                chain_tensor = chain_tensor[self.burn_in:]
            else:
                raise ValueError(f"Insufficient post-burn-in samples: total_samples={chain_tensor.shape[0]}, burn_in={self.burn_in}. Need at least {self.burn_in + 2} total samples.")
        
        if chain_tensor.shape[0] < 2:
            return 0.0
            
        # Compute squared jumps
        diff = chain_tensor[1:] - chain_tensor[:-1]
        squared_jumps = torch.sum(diff ** 2, dim=1)
        
        return torch.mean(squared_jumps).item()
    
    def _precompute_increments(self, total_steps):
        """Pre-compute all random increments for efficient proposal generation.
        
        Args:
            total_steps: Total number of increments to pre-compute (including burn-in)
        """
        print(f"Pre-computing {total_steps} random increments for GPU optimization...")
        
        # Generate all random increments at once: increments ~ N(0, I)
        raw_increments = torch.randn(total_steps, self.dim, device=self.device, dtype=torch.float32)
        
        # Apply covariance structure: increments = chol @ raw_increments^T
        self.precomputed_increments = torch.matmul(raw_increments, self.proposal_cov_chol.T)
        self.increment_index = 0
        
        print(f"Increments pre-computed. Memory allocated: {self.precomputed_increments.numel() * 4 / 1e6:.1f} MB") 