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
    - Optional batch processing for multiple independent proposals (non-standard)
    """
    
    def __init__(self, dim: int, var: float, 
                 target_dist: TorchTargetDistribution | TargetDistribution = None, 
                 symmetric: bool = True, 
                 beta: float = 1.0, beta_ladder: float = None, swap_acceptance_rate: float = None,
                 device: str = None, batch_size: int = 1024, pre_allocate_steps: int = None, standard_rwm: bool = True):
        """Initialize the GPU-accelerated RandomWalkMH algorithm.
        
        Args:
            dim: Dimension of the target distribution
            var: Proposal variance
            target_dist: Target distribution to sample from (TargetDistribution or TorchTargetDistribution)
            symmetric: Whether proposal distribution is symmetric
            beta: Temperature parameter (inverse)
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
            batch_size: Number of proposals to process in parallel (for non-standard mode)
            pre_allocate_steps: Pre-allocate memory for this many steps (None for dynamic).
            NOTE: This should be set to the number of samples you want to generate.
            standard_rwm: If True, use standard RWM (sequential). If False, use batch processing. 
            IMPORTANT NOTE: If you use batch processing, you are doing "multiple proposal" or "delayed update" MCMC.
            This is not the same as standard RWM.
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
        self.batch_size = batch_size
        self.standard_rwm = standard_rwm
        
        if standard_rwm:
            self.name = "RWM_GPU_Standard"
        else:
            self.name = "RWM_GPU_Batch"
        
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
            self.use_gpu_batch_target = False
            # Ensure target distribution is on the same device
            self.target_dist.to(self.device)
        elif hasattr(self.target_dist, 'batch_density_gpu'):
            # Legacy GPU target distribution
            self.use_torch_target = False
            self.use_gpu_batch_target = True
        else:
            # Legacy CPU target distribution
            self.use_torch_target = False
            self.use_gpu_batch_target = False
    
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
            self.batch_step(batch_size=1)
    
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
        
        if accept:
            self.current_state = proposal
            self.log_target_density_current = log_density_proposed.item()
            self.num_acceptances += 1
        
        # Add to chain and update statistics
        self._add_to_chain(self.current_state)
        self.total_steps += 1
        self.acceptance_rate = self.num_acceptances / self.total_steps
    
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
        elif self.use_gpu_batch_target:
            # Legacy GPU target distribution
            raise NotImplementedError("GPU target distribution not implemented")
            density_proposed = self.target_dist.batch_density_gpu(proposal.unsqueeze(0))[0]
            log_density_proposed = torch.log(density_proposed + 1e-300)
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
    
    def batch_step(self, batch_size=None):
        """Take multiple MCMC steps in parallel using vectorized operations.
        
        Args:
            batch_size: Number of proposals to process in parallel
        """
        if batch_size is None:
            batch_size = self.batch_size
            
        # Initialize current state if needed
        if self.current_state is None:
            # Use the initial sample from the base class (to match CPU behavior)
            self.current_state = torch.tensor(self.chain[-1], device=self.device, dtype=torch.float32)
            self.log_target_density_current = self._compute_log_density(self.current_state)
            
            # Add initial state to pre-allocated chain if using pre-allocation
            if self.pre_allocated_chain is not None and self.chain_index == 0:
                self.pre_allocated_chain[self.chain_index] = self.current_state
                self.chain_index += 1
        
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
        if (self.precomputed_increments is not None and 
            self.increment_index + batch_size <= self.precomputed_increments.shape[0]):
            # Use pre-computed increments for optimal GPU performance
            increments = self.precomputed_increments[self.increment_index:self.increment_index + batch_size]
            self.increment_index += batch_size
            proposals = self.current_state.unsqueeze(0) + increments
        else:
            # Fallback to on-demand generation (less efficient)
            noise = torch.randn(batch_size, self.dim, device=self.device, dtype=torch.float32)
            proposals = self.current_state.unsqueeze(0) + torch.matmul(noise, self.proposal_cov_chol.T)
        
        return proposals
    
    def _compute_batch_acceptance_ratios(self, proposals):
        """Compute log acceptance ratios for batch of proposals."""
        batch_size = proposals.shape[0]
        
        # Compute log densities for all proposals
        if self.use_gpu_batch_target:
            log_densities_proposed = torch.log(
                self.target_dist.batch_density_gpu(proposals) + 1e-300
            )
        else:
            densities_proposed = self._evaluate_target_density_cpu(proposals)
            log_densities_proposed = torch.log(torch.tensor(densities_proposed + 1e-300, device=self.device, dtype=torch.float32))
        
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
        elif self.use_gpu_batch_target:
            density = self.target_dist.batch_density_gpu(state.unsqueeze(0))[0]
            log_density = torch.log(density + 1e-300)
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
        """Generate samples using the selected method (standard or batch).
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            Chain of samples
        """
        if self.standard_rwm:
            return self.generate_samples_standard(num_samples)
        else:
            return self.generate_samples_batch(num_samples)
    
    def generate_samples_standard(self, num_samples):
        """Generate samples using standard sequential RWM with GPU acceleration.
        
        Args:
            num_samples: Total number of samples to generate
            
        Returns:
            Chain of samples as a torch tensor
        """
        print(f"Generating {num_samples} samples using standard RWM with GPU acceleration")
        
        # Pre-compute all increments for maximum GPU efficiency
        self._precompute_increments(num_samples)
        
        for i in range(num_samples):
            self._standard_step()
            
            if (i + 1) % 1000 == 0:
                print(f"Generated {i + 1}/{num_samples} samples "
                      f"(Accept rate: {self.acceptance_rate:.3f})")
        
        # Return the correct chain based on allocation method
        if self.pre_allocated_chain is not None:
            return self.pre_allocated_chain[:self.chain_index]
        else:
            return self.chain
    
    def generate_samples_batch(self, num_samples, batch_size=None):
        """Generate samples using batch processing for maximum efficiency.
        
        Args:
            num_samples: Total number of samples to generate
            batch_size: Batch size for parallel processing
            
        Returns:
            Chain of samples as a torch tensor
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
        
        # Return the correct chain based on allocation method
        if self.pre_allocated_chain is not None:
            return self.pre_allocated_chain[:self.chain_index]
        else:
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
    
    def _precompute_increments(self, num_samples):
        """Pre-compute all random increments for efficient proposal generation.
        
        Args:
            num_samples: Number of increments to pre-compute
        """
        print(f"Pre-computing {num_samples} random increments for GPU optimization...")
        
        # Generate all random increments at once: increments ~ N(0, I)
        raw_increments = torch.randn(num_samples, self.dim, device=self.device, dtype=torch.float32)
        
        # Apply covariance structure: increments = chol @ raw_increments^T
        self.precomputed_increments = torch.matmul(raw_increments, self.proposal_cov_chol.T)
        self.increment_index = 0
        
        print(f"Increments pre-computed. Memory allocated: {self.precomputed_increments.numel() * 4 / 1e6:.1f} MB") 