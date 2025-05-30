import torch
from typing import Optional, Union
from .base import ProposalDistribution

class LaplaceProposal(ProposalDistribution):
    """Multivariate Laplace proposal distribution.
    
    Optimized GPU implementation using inverse CDF transformation.
    """
    
    def __init__(self, dim: int, base_variance_vector: torch.Tensor, beta: float,
                 device: torch.device, dtype: torch.dtype,
                 rng_generator: Optional[torch.Generator] = None):
        """Initialize Laplace proposal.
        
        Args:
            base_variance_vector: Per-dimension variance vector (before beta scaling)
            Other args: See ProposalDistribution.__init__
        """
        super().__init__(dim, beta, device, dtype, rng_generator)
        self.name = "Laplace"
        
        if base_variance_vector.shape != (dim,):
            raise ValueError(f"base_variance_vector must have shape ({dim},), got {base_variance_vector.shape}")
        if not (base_variance_vector > 0).all():
            raise ValueError("All elements of base_variance_vector must be positive")
        
        # Move to correct device/dtype and scale by beta
        effective_variance_vector = base_variance_vector.to(device=self.device, dtype=self.dtype) / self.beta
        
        # For Laplace distribution: Var = 2 * scale^2, so scale = sqrt(Var / 2)
        self.scale_vector = torch.sqrt(effective_variance_vector / 2.0)
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate Laplace proposal increments using optimized inverse CDF."""
        return _sample_laplace_jit(n_samples, self.scale_vector, 
                                 self.device, self.dtype, self.rng_generator)
    
    def sample_into(self, n_samples: int, output_tensor: torch.Tensor) -> None:
        """Ultra-efficient in-place Laplace sampling."""
        _sample_laplace_into_jit(output_tensor, self.scale_vector, self.rng_generator)
    
    def get_name(self) -> str:
        return self.name

@torch.jit.script
def _sample_laplace_jit(n_samples: int, scale_vector: torch.Tensor,
                       device: torch.device, dtype: torch.dtype,
                       rng_generator: Optional[torch.Generator]) -> torch.Tensor:
    """JIT-compiled Laplace sampling using efficient inverse CDF transform."""
    dim = scale_vector.shape[0]
    
    # Generate U ~ Uniform(-0.5, 0.5) for inverse CDF
    if rng_generator is not None:
        u_rand = torch.rand((n_samples, dim), device=device, dtype=dtype, 
                          generator=rng_generator) - 0.5
    else:
        u_rand = torch.rand((n_samples, dim), device=device, dtype=dtype) - 0.5
    
    # Laplace inverse CDF: X = -scale * sign(U) * ln(1 - 2*|U|)
    # Use log1p for numerical stability: log1p(x) = log(1 + x)
    abs_u = torch.abs(u_rand)
    sign_u = torch.sign(u_rand)
    
    # Clamp to avoid log(0) due to floating point precision
    clamped_arg = torch.clamp(-2.0 * abs_u, min=-0.999999)
    samples = -scale_vector.unsqueeze(0) * sign_u * torch.log1p(clamped_arg)
    
    return samples


@torch.jit.script
def _sample_laplace_into_jit(output_tensor: torch.Tensor, scale_vector: torch.Tensor,
                            rng_generator: Optional[torch.Generator]) -> None:
    """JIT-compiled in-place Laplace sampling."""
    n_samples, dim = output_tensor.shape
    
    # Generate uniform random values in-place
    if rng_generator is not None:
        output_tensor.uniform_(-0.5, 0.5, generator=rng_generator)
    else:
        output_tensor.uniform_(-0.5, 0.5)
    
    # Apply Laplace transform in-place
    abs_u = torch.abs(output_tensor)
    sign_u = torch.sign(output_tensor)
    
    # Transform: X = -scale * sign(U) * ln(1 - 2*|U|)
    abs_u.mul_(-2.0).clamp_(min=-0.999999)
    torch.log1p(abs_u, out=abs_u)
    output_tensor = -scale_vector.unsqueeze(0) * sign_u * abs_u