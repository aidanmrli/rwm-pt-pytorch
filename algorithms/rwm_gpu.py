import torch
import torch.nn as nn
import numpy as np
from interfaces import MHAlgorithm, TargetDistribution
import warnings

class RandomWalkMH_GPU(MHAlgorithm):
    """GPU-accelerated vectorized implementation of Random Walk Metropolis-Hastings algorithm.
    
    This implementation provides significant performance improvements through:
    - Batch processing of multiple proposals
    - GPU acceleration via PyTorch
    - Pre-allocated memory for chains
    - Vectorized density evaluations
    - Reduced redundant computations
    """
    
    def __init__(self, dim, var, target_dist: TargetDistribution = None, 
                 symmetric=True, beta=1.0, beta_ladder=None, swap_acceptance_rate=None,
                 device=None, batch_size=1024, pre_allocate_steps=None):
        """Initialize the GPU-accelerated RandomWalkMH algorithm.
        
        Args:
            dim: Dimension of the target distribution
            var: Proposal variance
            target_dist: Target distribution to sample from
            symmetric: Whether proposal distribution is symmetric
            beta: Temperature parameter (inverse)
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
            batch_size: Number of proposals to process in parallel
            pre_allocate_steps: Pre-allocate memory for this many steps (None for dynamic)
        """
        super().__init__(dim, var, target_dist, symmetric)
        
        # GPU setup
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
            
        if self.device.type == 'cuda':
            print(f"Using GPU acceleration: {torch.cuda.get_device_name()}")
        else:
            print("Using CPU (consider installing CUDA for better performance)")
        
        # Algorithm parameters
        self.beta = beta
        self.batch_size = batch_size
        self.name = "RWM_GPU"
        
        # Performance tracking
        self.num_acceptances = 0
        self.acceptance_rate = 0
        self.total_steps = 0
        
        # Pre-allocate memory if requested
        self.pre_allocate_steps = pre_allocate_steps
        if pre_allocate_steps:
            self.pre_allocated_chain = torch.zeros(
                (pre_allocate_steps + 1, dim), 
                device=self.device, 
                dtype=torch.float32
            )
            self.chain_index = 0
        else:
            self.pre_allocated_chain = None
            self.chain_index = None
            
        # Convert proposal covariance to GPU
        self.proposal_cov = (var / beta) * torch.eye(dim, device=self.device, dtype=torch.float32)
        self.proposal_cov_chol = torch.linalg.cholesky(self.proposal_cov)
        
        # Current state tracking
        self.current_state = None
        self.log_target_density_current = -float('inf')
        
        # Cache for target distribution if it supports GPU
        self._setup_target_distribution_cache()
    
    def _setup_target_distribution_cache(self):
        """Setup GPU-compatible target distribution evaluation."""
        # Check if target distribution supports batch evaluation
        if hasattr(self.target_dist, 'batch_density_gpu'):
            self.use_gpu_target = True
        else:
            self.use_gpu_target = False
            # Create a vectorized wrapper for CPU target distributions
            self._cached_target_density = self._vectorized_target_density
    
    def _vectorized_target_density(self, states_tensor):
        """Vectorized evaluation of target density for CPU distributions."""
        states_np = states_tensor.cpu().numpy()
        if len(states_np.shape) == 1:
            states_np = states_np.reshape(1, -1)
        
        densities = np.zeros(states_np.shape[0])
        for i, state in enumerate(states_np):
            densities[i] = self.target_density(state)
        
        return torch.tensor(densities, device=self.device, dtype=torch.float32)
    
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
    
    def step(self):
        """Take a single MCMC step."""
        self.batch_step(batch_size=1)
    
    def batch_step(self, batch_size=None):
        """Take multiple MCMC steps in parallel using vectorized operations.
        
        Args:
            batch_size: Number of proposals to process in parallel
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Initialize current state if needed
        if self.current_state is None:
            if len(self.chain) == 0:
                # Initialize with a random state
                self.current_state = torch.randn(self.dim, device=self.device, dtype=torch.float32)
                self._add_to_chain(self.current_state)
                self.log_target_density_current = self._compute_log_density(self.current_state)
            else:
                self.current_state = torch.tensor(self.chain[-1], device=self.device, dtype=torch.float32)
                self.log_target_density_current = self._compute_log_density(self.current_state)
        
        # Generate batch of proposals
        proposals = self._generate_proposals(batch_size)
        
        # Compute acceptance probabilities
        log_accept_ratios, log_densities = self._compute_batch_acceptance_ratios(proposals)
        
        # Make acceptance decisions
        accept_flags = self._make_acceptance_decisions(log_accept_ratios)
        
        # Process acceptances sequentially (MCMC is inherently sequential)
        for i in range(batch_size):
            if accept_flags[i]:
                self.current_state = proposals[i].clone()
                self.log_target_density_current = log_densities[i].item()
                self.num_acceptances += 1
            
            self._add_to_chain(self.current_state)
            self.total_steps += 1
            self.acceptance_rate = self.num_acceptances / self.total_steps
    
    def _generate_proposals(self, batch_size):
        """Generate batch of proposal states."""
        # Generate random noise
        noise = torch.randn(batch_size, self.dim, device=self.device, dtype=torch.float32)
        
        # Apply covariance structure: proposals = current + chol @ noise^T
        proposals = self.current_state.unsqueeze(0) + torch.matmul(noise, self.proposal_cov_chol.T)
        
        return proposals
    
    def _compute_batch_acceptance_ratios(self, proposals):
        """Compute log acceptance ratios for batch of proposals."""
        batch_size = proposals.shape[0]
        
        # Compute log densities for all proposals
        if self.use_gpu_target:
            log_densities_proposed = torch.log(
                self.target_dist.batch_density_gpu(proposals) + 1e-300
            )
        else:
            densities_proposed = self._cached_target_density(proposals)
            log_densities_proposed = torch.log(densities_proposed + 1e-300)
        
        # Compute acceptance ratios
        log_accept_ratios = self.beta * (log_densities_proposed - self.log_target_density_current)
        
        if not self.symmetric:
            # Add proposal ratio terms (currently assuming symmetric proposals)
            warnings.warn("Asymmetric proposals not yet optimized for GPU implementation")
        
        return log_accept_ratios, log_densities_proposed
    
    def _make_acceptance_decisions(self, log_accept_ratios):
        """Make vectorized acceptance/rejection decisions."""
        # Generate random numbers for acceptance decisions
        random_vals = torch.rand(log_accept_ratios.shape[0], device=self.device)
        
        # Accept if log_ratio > 0 or random < exp(log_ratio)
        accept_flags = (log_accept_ratios > 0) | (random_vals < torch.exp(log_accept_ratios))
        
        return accept_flags
    
    def _compute_log_density(self, state):
        """Compute log density for a single state."""
        if self.use_gpu_target:
            density = self.target_dist.batch_density_gpu(state.unsqueeze(0))[0]
        else:
            density = self._cached_target_density(state.unsqueeze(0))[0]
        
        return torch.log(density + 1e-300).item()
    
    def _add_to_chain(self, state):
        """Add state to the chain (either pre-allocated or dynamic)."""
        if self.pre_allocated_chain is not None:
            if self.chain_index < self.pre_allocated_chain.shape[0]:
                self.pre_allocated_chain[self.chain_index] = state
                self.chain_index += 1
            else:
                warnings.warn("Pre-allocated chain full, switching to dynamic allocation")
                self.pre_allocated_chain = None
                self.chain.append(state.cpu().numpy())
        else:
            self.chain.append(state.cpu().numpy())
    
    def get_chain_gpu(self):
        """Get the chain as a GPU tensor."""
        if self.pre_allocated_chain is not None:
            return self.pre_allocated_chain[:self.chain_index]
        else:
            return torch.tensor(np.array(self.chain), device=self.device, dtype=torch.float32)
    
    def generate_samples_batch(self, num_samples, batch_size=None):
        """Generate samples using batch processing for maximum efficiency.
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size for parallel processing
            
        Returns:
            Chain of samples
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Calculate number of batches needed
        num_batches = (num_samples + batch_size - 1) // batch_size
        
        print(f"Generating {num_samples} samples using {num_batches} batches of size {batch_size}")
        
        for batch_idx in range(num_batches):
            remaining_samples = min(batch_size, num_samples - batch_idx * batch_size)
            self.batch_step(remaining_samples)
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Completed batch {batch_idx + 1}/{num_batches} "
                      f"(Accept rate: {self.acceptance_rate:.3f})")
        
        return self.chain
    
    def expected_squared_jump_distance_gpu(self):
        """Compute ESJD using GPU tensors for efficiency."""
        if self.pre_allocated_chain is not None and self.chain_index > 1:
            chain_tensor = self.pre_allocated_chain[:self.chain_index]
        else:
            chain_tensor = self.get_chain_gpu()
        
        if chain_tensor.shape[0] < 2:
            return 0.0
            
        # Compute squared jumps
        diff = chain_tensor[1:] - chain_tensor[:-1]
        squared_jumps = torch.sum(diff ** 2, dim=1)
        
        return torch.mean(squared_jumps).item() 