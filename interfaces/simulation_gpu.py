import numpy as np
import torch
from scipy.stats import norm
import matplotlib.pyplot as plt
from typing import Optional, Callable, Union
import tqdm
import time
from .metropolis import MHAlgorithm
from .target import TargetDistribution
from .target_torch import TorchTargetDistribution
from proposal_distributions import ProposalDistribution, NormalProposal, LaplaceProposal, UniformRadiusProposal

class MCMCSimulation_GPU:
    """
    GPU-accelerated MCMC simulation framework with comprehensive performance optimization.
    
    This class provides a high-level interface for running MCMC simulations on GPU with
    automatic device management, performance benchmarking, and result analysis.
    """
    
    def __init__(self, 
                 dim: int, 
                 sigma: float = None,  # Backward compatibility
                 proposal_config: dict = None,  # New proposal system
                 num_iterations: int = 1000, 
                 algorithm: MHAlgorithm = None,
                 target_dist: Union[TargetDistribution, TorchTargetDistribution] = None,
                 symmetric: bool = True,
                 seed: Optional[int] = None,
                 beta_ladder: Optional[list] = None,
                 swap_acceptance_rate: Optional[float] = None,
                 device: Optional[str] = None,
                 pre_allocate: bool = True,
                 burn_in: int = 0,
                 **kwargs):
        """
        Initialize GPU-accelerated MCMC simulation.
        
        Args:
            dim: Dimension of the target distribution
            sigma: Proposal variance (backward compatibility - creates NormalProposal)
            proposal_config: Configuration dict for proposal distribution:
                {'name': 'Normal', 'params': {'base_variance_scalar': 0.1}}
                {'name': 'Laplace', 'params': {'base_variance_vector': [0.1, 0.2, ...]}}
                {'name': 'UniformRadius', 'params': {'base_radius': 1.0}}
            num_iterations: Number of MCMC iterations
            algorithm: MCMC algorithm class
            target_dist: Target distribution
            symmetric: Whether proposal is symmetric
            seed: Random seed
            beta_ladder: Temperature ladder for parallel tempering
            swap_acceptance_rate: Target swap acceptance rate for PT
            device: GPU device to use ('cuda', 'cpu', or None for auto-detection)
            pre_allocate: Whether to pre-allocate GPU memory for chains
            burn_in: Number of initial samples to discard for MCMC burn-in (default: 0)
            **kwargs: Additional algorithm-specific parameters
        """
        # Handle backward compatibility and proposal configuration
        if proposal_config is None and sigma is not None:
            # Backward compatibility: create Normal proposal from sigma
            proposal_config = {
                'name': 'Normal',
                'params': {'base_variance_scalar': sigma}
            }
        elif proposal_config is None and sigma is None:
            raise ValueError("Either sigma (backward compatibility) or proposal_config must be provided")
        
        self.num_iterations = num_iterations
        self.burn_in = max(0, min(burn_in, num_iterations - 1))
        self.target_dist = target_dist
        self.proposal_config = proposal_config
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Set device
        self.device = device
        self.pre_allocate = pre_allocate
        
        # Determine beta for proposal (for algorithms that use it)
        if hasattr(algorithm, '__name__') and 'GPU' in algorithm.__name__:
            # Check if this is a parallel tempering algorithm
            if 'ParallelTempering' in algorithm.__name__:
                # For PT, each chain has its own beta from beta_ladder
                algo_beta = 1.0  # PT handles beta internally
                self.algorithm = algorithm(
                    dim, sigma, target_dist, symmetric, 
                    device=device, 
                    pre_allocate_steps=num_iterations if pre_allocate else None,
                    beta_ladder=beta_ladder,
                    swap_acceptance_rate=swap_acceptance_rate,
                    burn_in=self.burn_in,
                    **kwargs
                )
            else:
                # Regular GPU RWM algorithm - create proposal distribution
                algo_beta = beta_ladder[0] if beta_ladder else 1.0
                
                # Create proposal distribution if proposals are available
                if ProposalDistribution is not None:
                    proposal_dist = self._create_proposal_distribution(
                        dim=dim, 
                        beta=algo_beta, 
                        proposal_config=proposal_config,
                        device=torch.device(device),
                        dtype=torch.float32,
                        use_efficient_rng=kwargs.get('use_efficient_rng', True)
                    )
                    
                    # Create algorithm with proposal distribution
                    self.algorithm = algorithm(
                        dim=dim,
                        proposal_distribution=proposal_dist,
                        target_dist=target_dist,
                        symmetric=symmetric,
                        beta=algo_beta,
                        device=device,
                        pre_allocate_steps=num_iterations if pre_allocate else None,
                        burn_in=self.burn_in,
                        use_efficient_rng=kwargs.get('use_efficient_rng', True),
                        **{k: v for k, v in kwargs.items() if k != 'use_efficient_rng'}
                    )
                else:
                    # Fallback to old sigma-based approach
                    self.algorithm = algorithm(
                        dim, sigma, target_dist, symmetric, 
                        device=device, 
                        pre_allocate_steps=num_iterations if pre_allocate else None,
                        beta=algo_beta,
                        burn_in=self.burn_in,
                        **kwargs
                    )
        else:
            # Standard algorithm - use old interface
            self.algorithm = algorithm(
                dim, sigma, target_dist, symmetric, 
                beta_ladder=beta_ladder, 
                swap_acceptance_rate=swap_acceptance_rate,
                burn_in=self.burn_in,
                **kwargs
            )
        
        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed)
        
        self._start_time = None
        self._end_time = None
    
    def reset(self):
        """Reset the simulation to the initial state."""
        self.algorithm.reset()

    def has_run(self):
        """Return whether the algorithm has been run."""
        # Check if using pre-allocated chain
        if hasattr(self.algorithm, 'pre_allocated_chain') and self.algorithm.pre_allocated_chain is not None:
            return self.algorithm.chain_index > 1
        else:
            return len(self.algorithm.chain) > 1

    def generate_samples(self, progress_bar=True):
        """
        Generate samples step-by-step.
        
        Args:
            progress_bar: Whether to show progress bar
            
        Returns:
            Chain of samples
        """
        if self.has_run():
            raise ValueError("Please reset the algorithm before running it again.")
        
        start_time = time.time()
        
        # Check if this is a GPU algorithm with its own optimized generate_samples method
        if hasattr(self.algorithm, 'generate_samples') and ('ParallelTempering' in self.algorithm.__class__.__name__ or 'GPU' in self.algorithm.__class__.__name__):
            # Use the algorithm's own optimized generate_samples method
            if 'ParallelTempering' in self.algorithm.__class__.__name__:
                print(f"Using optimized GPU parallel tempering generate_samples method")
            else:
                print(f"Using optimized GPU {self.algorithm.__class__.__name__} generate_samples method")
            chain = self.algorithm.generate_samples(self.num_iterations)
            # Convert to list format for compatibility
            if hasattr(chain, 'cpu'):
                chain = chain.cpu().numpy().tolist()
            elif isinstance(chain, torch.Tensor):
                chain = chain.tolist()
        else:
            # Use step-by-step interface for other algorithms
            if progress_bar:
                with tqdm.tqdm(total=self.num_iterations, desc="Running MCMC", unit="iteration") as pbar:
                    for i in range(self.num_iterations + self.burn_in):
                        self.algorithm.step()
                        pbar.update(1)
            else:
                for i in range(self.num_iterations + self.burn_in):
                    self.algorithm.step()
                    
            chain = self.algorithm.chain
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"Drew {self.num_iterations} samples in {elapsed_time:.2f} seconds ({self.num_iterations/elapsed_time:.0f} samples/s)")
        print(f"Final acceptance rate: {self.acceptance_rate():.3f}")
        
        return chain
    
    def acceptance_rate(self):
        """Return the acceptance rate of the algorithm, excluding burn-in samples."""
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        # The burn-in is already excluded in the acceptance rate calculation
        return self.algorithm.acceptance_rate

    def expected_squared_jump_distance(self):
        """
        Calculate the expected squared jump distance using optimized GPU computation if available,
        excluding burn-in samples.
        """
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        
        # Use GPU-accelerated ESJD computation if available.
        # The burn-in is already excluded in the GPU ESJD calculation.
        if hasattr(self.algorithm, 'expected_squared_jump_distance_gpu'):
            return self.algorithm.expected_squared_jump_distance_gpu()
        else:
            # Fall back to CPU computation
            chain = np.array(self.algorithm.chain)
            if self.burn_in > 0 and len(chain) > self.burn_in + 1:
                chain_post_burnin = chain[self.burn_in:]
                squared_jumps = np.sum((chain_post_burnin[1:] - chain_post_burnin[:-1]) ** 2, axis=1)
                return np.mean(squared_jumps)
            elif self.burn_in == 0:
                squared_jumps = np.sum((chain[1:] - chain[:-1]) ** 2, axis=1)
                return np.mean(squared_jumps)
            else:
                return 0.0
    
    def pt_expected_squared_jump_distance(self):
        """Calculate the expected squared jump distance for parallel tempering."""
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        return self.algorithm.pt_esjd

    def benchmark_performance(self, num_samples_list=[1000, 5000, 10000, 50000], 
                             compare_cpu=True):
        """
        Benchmark performance across different sample sizes and compare GPU vs CPU.
        
        Args:
            num_samples_list: List of sample sizes to benchmark
            compare_cpu: Whether to compare with CPU implementation
            
        Returns:
            Dictionary with benchmark results
        """
        results = {
            'sample_sizes': num_samples_list,
            'gpu_times': [],
            'gpu_samples_per_sec': [],
            'cpu_times': [] if compare_cpu else None,
            'cpu_samples_per_sec': [] if compare_cpu else None,
            'speedup': [] if compare_cpu else None
        }
        
        for num_samples in num_samples_list:
            print(f"\nBenchmarking {num_samples} samples...")
            
            # Reset for each test
            self.reset()
            original_iterations = self.num_iterations
            self.num_iterations = num_samples
            
            # GPU benchmark
            start_time = time.time()
            self.generate_samples(progress_bar=False)
            gpu_time = time.time() - start_time
            gpu_sps = num_samples / gpu_time
            
            results['gpu_times'].append(gpu_time)
            results['gpu_samples_per_sec'].append(gpu_sps)
            
            print(f"GPU: {gpu_time:.2f}s, {gpu_sps:.0f} samples/sec")
            
            if compare_cpu:
                # CPU benchmark (if available)
                self.reset()
                start_time = time.time()
                self.generate_samples(progress_bar=False)
                cpu_time = time.time() - start_time
                cpu_sps = num_samples / cpu_time
                speedup = cpu_time / gpu_time
                
                results['cpu_times'].append(cpu_time)
                results['cpu_samples_per_sec'].append(cpu_sps)
                results['speedup'].append(speedup)
                
                print(f"CPU: {cpu_time:.2f}s, {cpu_sps:.0f} samples/sec")
                print(f"Speedup: {speedup:.1f}x")
            
            # Restore original iterations
            self.num_iterations = original_iterations
        
        return results

    def traceplot(self, single_dim=False, show=False, use_gpu_data=True):
        """
        Create traceplots with optional GPU tensor optimization.
        """
        plt.figure(figsize=(10, 6))
        if not self.has_run():
            raise ValueError("The algorithm has not been run yet.")
        
        # Get chain data (potentially from GPU)
        if use_gpu_data and hasattr(self.algorithm, 'get_chain_gpu'):
            chain = self.algorithm.get_chain_gpu().cpu().numpy()
        else:
            chain = np.array(self.algorithm.chain)
        
        if single_dim:
            plt.plot(chain[:, 0], label=f"Dimension 1", alpha=0.7, lw=0.5)
        else:
            for i in range(min(5, self.algorithm.dim)):  # Limit to 5 dimensions for readability
                plt.plot(chain[:, i], label=f"Dimension {i + 1}", alpha=0.7, lw=0.5)

        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
        plt.title(f'Traceplot - {self.algorithm.get_name()} (GPU-accelerated)')
        
        filename = f"images/traceplot_{self.target_dist.get_name()}_{self.algorithm.get_name()}_dim{self.algorithm.dim}_{self.num_iterations}iters"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.clf()

    def samples_histogram(self, num_bins=50, axis=0, show=False, use_gpu_data=True):
        """
        Create sample histograms with optional GPU tensor optimization.
        """
        plt.figure(figsize=(10, 6))
        
        # Get chain data (potentially from GPU)
        if use_gpu_data and hasattr(self.algorithm, 'get_chain_gpu'):
            chain_tensor = self.algorithm.get_chain_gpu()
            samples = chain_tensor[:, axis].cpu().numpy()
        else:
            samples = np.array(self.algorithm.chain)[:, axis]
        
        plt.hist(samples, bins=num_bins, density=True, alpha=0.5, label='Samples')

        # Generate values for plotting the target density
        x = np.linspace(min(-20, min(samples) - 2), max(20, max(samples) + 2), 1000)
        y = np.zeros_like(x)
        
        for i in range(len(x)):
            y[i] = self.target_dist.density(x[i])

        plt.plot(x, y, color='red', linestyle='--', linewidth=2, label='Target Density')
        plt.xlabel('Value')
        plt.ylabel('Density')
        plt.legend()
        plt.title(f'Sample Histogram - {self.algorithm.get_name()} (GPU-accelerated)')
        
        filename = f"images/hist_gpu_{self.target_dist.get_name()}_{self.algorithm.get_name()}_dim{self.algorithm.dim}_{self.num_iterations}iters"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        plt.clf()

    def _create_proposal_distribution(self, dim: int, beta: float, proposal_config: dict, 
                                     device: torch.device, dtype: torch.dtype, 
                                     use_efficient_rng: bool = True) -> 'ProposalDistribution':
        """Create a proposal distribution object from configuration.
        
        Args:
            dim: Dimension of the distribution
            beta: Inverse temperature parameter
            proposal_config: Configuration dictionary
            device: PyTorch device
            dtype: Data type for tensors
            use_efficient_rng: Whether to use efficient RNG
            
        Returns:
            ProposalDistribution instance
        """
        if ProposalDistribution is None:
            raise ImportError("Proposal distribution classes not available. Please check algorithms.proposals import.")
        
        name = proposal_config.get('name')
        params = proposal_config.get('params', {})
        
        # Setup RNG generator if requested and on CUDA
        rng_generator = None
        if use_efficient_rng and device.type == 'cuda':
            rng_generator = torch.Generator(device=device)
        
        if name == "Normal":
            base_variance_scalar = params.get('base_variance_scalar')
            if base_variance_scalar is None:
                raise ValueError("Normal proposal requires 'base_variance_scalar' parameter")
            return NormalProposal(dim, base_variance_scalar, beta, device, dtype, rng_generator)
        
        elif name == "Laplace":
            base_variance_vector = params.get('base_variance_vector')
            if base_variance_vector is None:
                raise ValueError("Laplace proposal requires 'base_variance_vector' parameter")
            
            # Convert to tensor if needed
            if isinstance(base_variance_vector, (list, tuple)):
                base_variance_vector = torch.tensor(base_variance_vector, dtype=dtype)
            elif isinstance(base_variance_vector, (int, float)):
                # Scalar -> isotropic vector
                base_variance_vector = torch.full((dim,), float(base_variance_vector), dtype=dtype)
            elif isinstance(base_variance_vector, torch.Tensor):
                base_variance_vector = base_variance_vector.to(dtype=dtype)
            else:
                raise ValueError(f"Invalid base_variance_vector type: {type(base_variance_vector)}")
            
            return LaplaceProposal(dim, base_variance_vector, beta, device, dtype, rng_generator)
        
        elif name == "UniformRadius":
            base_radius = params.get('base_radius')
            if base_radius is None:
                raise ValueError("UniformRadius proposal requires 'base_radius' parameter")
            return UniformRadiusProposal(dim, base_radius, beta, device, dtype, rng_generator)
        
        else:
            raise ValueError(f"Unknown proposal distribution name: {name}") 