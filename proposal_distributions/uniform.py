import torch
from typing import Optional
from .base import ProposalDistribution


class UniformRadiusProposal(ProposalDistribution):
    """Uniform distribution over n-dimensional ball.
    
    Optimized GPU implementation using direction+radius sampling.
    """
    
    def __init__(self, dim: int, base_radius: float, beta: float,
                 device: torch.device, dtype: torch.dtype,
                 rng_generator: Optional[torch.Generator] = None):
        """Initialize Uniform radius proposal.
        
        Args:
            base_radius: Base radius (before beta scaling)
            Other args: See ProposalDistribution.__init__
        """
        super().__init__(dim, beta, device, dtype, rng_generator)
        self.name = "UniformRadius"
        
        if base_radius <= 0:
            raise ValueError("base_radius must be positive")
        
        # Scale radius by inverse square root of beta (heuristic scaling)
        self.effective_radius = base_radius / torch.sqrt(torch.tensor(self.beta, 
                                                                    device=self.device, dtype=self.dtype))
        
        # Pre-compute inverse dimension for efficient radius scaling
        self.inv_dim = 1.0 / self.dim
    
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate uniform n-ball samples using optimized direction+radius method."""
        return _sample_uniform_ball_jit(n_samples, self.dim, self.effective_radius, 
                                      self.inv_dim, self.device, self.dtype, self.rng_generator)
    
    def sample_into(self, n_samples: int, output_tensor: torch.Tensor) -> None:
        """Ultra-efficient in-place uniform ball sampling."""
        _sample_uniform_ball_into_jit(output_tensor, self.effective_radius, 
                                    self.inv_dim, self.rng_generator)
    
    def get_name(self) -> str:
        return self.name
    
@torch.jit.script
def _sample_uniform_ball_jit(n_samples: int, dim: int, effective_radius: torch.Tensor,
                            inv_dim: float, device: torch.device, dtype: torch.dtype,
                            rng_generator: Optional[torch.Generator]) -> torch.Tensor:
    """JIT-compiled uniform n-ball sampling using direction+radius method."""
    
    # Step 1: Generate random directions using Gaussian sampling + normalization
    if rng_generator is not None:
        directions = torch.randn((n_samples, dim), device=device, dtype=dtype, generator=rng_generator)
    else:
        directions = torch.randn((n_samples, dim), device=device, dtype=dtype)
    
    # Normalize to unit sphere (handle zero vectors for numerical safety)
    norms = torch.linalg.norm(directions, dim=1, keepdim=True)
    safe_norms = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
    directions.div_(safe_norms)
    
    # Step 2: Generate random radii using inverse CDF: R = R_max * U^(1/d)
    if rng_generator is not None:
        u_uniform = torch.rand((n_samples, 1), device=device, dtype=dtype, generator=rng_generator)
    else:
        u_uniform = torch.rand((n_samples, 1), device=device, dtype=dtype)
    
    radii = effective_radius * torch.pow(u_uniform, inv_dim)
    
    # Step 3: Scale directions by radii
    return directions * radii


@torch.jit.script
def _sample_uniform_ball_into_jit(output_tensor: torch.Tensor, effective_radius: torch.Tensor,
                                 inv_dim: float, rng_generator: Optional[torch.Generator]) -> None:
    """JIT-compiled in-place uniform n-ball sampling."""
    n_samples, dim = output_tensor.shape
    
    # Generate directions in-place
    if rng_generator is not None:
        output_tensor.normal_(generator=rng_generator)
    else:
        output_tensor.normal_()
    
    # Normalize directions
    norms = torch.linalg.norm(output_tensor, dim=1, keepdim=True)
    safe_norms = torch.where(norms > 1e-12, norms, torch.ones_like(norms))
    output_tensor.div_(safe_norms)
    
    # Generate radii and scale
    if rng_generator is not None:
        radii = torch.rand((n_samples, 1), device=output_tensor.device, 
                         dtype=output_tensor.dtype, generator=rng_generator)
    else:
        radii = torch.rand((n_samples, 1), device=output_tensor.device, dtype=output_tensor.dtype)
    
    radii = effective_radius * torch.pow(radii, inv_dim)
    output_tensor.mul_(radii) 