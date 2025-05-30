import torch
from typing import Optional
from .base import ProposalDistribution

class NormalProposal(ProposalDistribution):
    """Multivariate Normal proposal distribution with diagonal covariance.
    
    Ultra-optimized for GPU with JIT compilation and efficient batch generation.
    """
    
    def __init__(self, dim: int, base_variance_scalar: float, beta: float, 
                 device: torch.device, dtype: torch.dtype, 
                 rng_generator: Optional[torch.Generator] = None):
        """Initialize Normal proposal.
        
        Args:
            base_variance_scalar: Base variance (before beta scaling)
            Other args: See ProposalDistribution.__init__
        """
        super().__init__(dim, beta, device, dtype, rng_generator)
        self.name = "Normal"
        
        if base_variance_scalar <= 0:
            raise ValueError("base_variance_scalar must be positive")
        
        # Effective variance scaled by beta (higher beta = smaller proposals)
        effective_variance = base_variance_scalar / self.beta
        
        # Pre-compute standard deviation for efficient sampling
        self.std_dev = torch.sqrt(torch.tensor(effective_variance, 
                                             device=self.device, dtype=self.dtype))
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate Normal proposal increments with maximum GPU efficiency."""
        return _sample_normal_jit(n_samples, self.dim, self.std_dev, 
                                self.device, self.dtype, self.rng_generator)
    
    def sample_into(self, n_samples: int, output_tensor: torch.Tensor) -> None:
        """Ultra-efficient in-place Normal sampling."""
        _sample_normal_into_jit(output_tensor, self.std_dev, self.rng_generator)
    
    def get_name(self) -> str:
        return self.name

# JIT-compiled sampling functions for maximum GPU performance
@torch.jit.script
def _sample_normal_jit(n_samples: int, dim: int, std_dev: torch.Tensor, 
                      device: torch.device, dtype: torch.dtype, 
                      rng_generator: Optional[torch.Generator]) -> torch.Tensor:
    """JIT-compiled Normal sampling for maximum performance."""
    if rng_generator is not None:
        noise = torch.randn((n_samples, dim), device=device, dtype=dtype, generator=rng_generator)
    else:
        noise = torch.randn((n_samples, dim), device=device, dtype=dtype)
    return noise * std_dev


@torch.jit.script  
def _sample_normal_into_jit(output_tensor: torch.Tensor, std_dev: torch.Tensor,
                           rng_generator: Optional[torch.Generator]) -> None:
    """JIT-compiled in-place Normal sampling."""
    if rng_generator is not None:
        output_tensor.normal_(generator=rng_generator)
    else:
        output_tensor.normal_()
    output_tensor.mul_(std_dev)