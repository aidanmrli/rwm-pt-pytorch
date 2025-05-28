import torch
import torch.nn as nn
from abc import ABC, abstractmethod

class TorchTargetDistribution(ABC):
    """PyTorch-native interface for target distributions with GPU acceleration."""

    def __init__(self, dimension, device=None):
        """Initialize the target distribution.
        
        Args:
            dimension: Dimensionality of the distribution
            device: PyTorch device ('cuda', 'cpu', or None for auto-detect)
        """
        self.dim = dimension
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

    @abstractmethod
    def density(self, x):
        """Compute the density of the distribution at point(s) x.
        
        Args:
            x: PyTorch tensor of shape (dim,) for single point or (batch_size, dim) for batch
            
        Returns:
            PyTorch tensor of densities with shape () for single point or (batch_size,) for batch
        """
        raise NotImplementedError("Subclasses must implement the density method.")

    @abstractmethod
    def log_density(self, x):
        """Compute the log density of the distribution at point(s) x.
        
        Args:
            x: PyTorch tensor of shape (dim,) for single point or (batch_size, dim) for batch
            
        Returns:
            PyTorch tensor of log densities with shape () for single point or (batch_size,) for batch
        """
        raise NotImplementedError("Subclasses must implement the log_density method.")

    @abstractmethod
    def get_name(self):
        """Return the name of the distribution as a string."""
        raise NotImplementedError("Subclasses must implement the get_name method.")

    def draw_sample(self, beta=1.0):
        """Draw a sample from the target distribution (CPU implementation for compatibility).
        
        This is meant to be a cheap heuristic used for constructing the temperature ladder 
        in parallel tempering. Do not use this to draw samples in an actual Metropolis algorithm.
        
        Args:
            beta: Inverse temperature parameter
            
        Returns:
            numpy.ndarray: A sample from the target distribution
        """
        raise NotImplementedError("Subclasses should implement draw_sample for compatibility.")

    def to(self, device):
        """Move the distribution to a specific device."""
        self.device = torch.device(device)
        return self 