import torch
import torch.nn as nn
from abc import ABC, abstractmethod
import warnings
from typing import Optional

class ProposalDistribution(ABC):
    """Abstract base class for proposal distributions in MCMC algorithms.
    
    Designed for maximum GPU performance with JIT compilation support,
    batch operations, and efficient memory management.
    """
    
    def __init__(self, dim: int, beta: float, device: torch.device, dtype: torch.dtype, 
                 rng_generator: Optional[torch.Generator] = None):
        """Initialize proposal distribution.
        
        Args:
            dim: Dimension of the proposal distribution
            beta: Inverse temperature for scaling (higher beta = smaller proposals)
            device: PyTorch device for computations
            dtype: Data type for tensors
            rng_generator: Optional RNG generator for reproducibility
        """
        self.dim = dim
        self.beta = beta
        self.device = device
        self.dtype = dtype
        self.rng_generator = rng_generator
    
    @abstractmethod
    def sample(self, n_samples: int) -> torch.Tensor:
        """Generate n_samples proposal increments.
        
        Args:
            n_samples: Number of proposal increments to generate
            
        Returns:
            Tensor of shape (n_samples, dim) with proposal increments
        """
        pass
    
    @abstractmethod
    def get_name(self) -> str:
        """Return the name of the proposal distribution."""
        pass
    
    def sample_into(self, n_samples: int, output_tensor: torch.Tensor) -> None:
        """Generate samples directly into pre-allocated tensor for memory efficiency.
        
        Args:
            n_samples: Number of samples to generate
            output_tensor: Pre-allocated tensor of shape (n_samples, dim)
        """
        # Default implementation - subclasses can override for better performance
        samples = self.sample(n_samples)
        output_tensor.copy_(samples) 