import torch
import numpy as np
from interfaces.target_torch import TorchTargetDistribution

class HypercubeTorch(TorchTargetDistribution):
    """
    PyTorch-native hypercube uniform distribution with full GPU acceleration.
    No CPU-GPU transfers during density evaluation.
    """

    def __init__(self, dim, left_boundary=0.0, right_boundary=1.0, device=None):
        """
        Initialize the PyTorch-native Hypercube distribution.

        Args:
            dim: Dimension of the distribution
            left_boundary: Left boundary of the hypercube (defaults to 0.0)
            right_boundary: Right boundary of the hypercube (defaults to 1.0)
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim, device)
        self.name = "HypercubeTorch"
        
        # Store boundaries as tensors on the device
        self.left_boundary = torch.tensor(left_boundary, device=self.device, dtype=torch.float32)
        self.right_boundary = torch.tensor(right_boundary, device=self.device, dtype=torch.float32)
        
        # Pre-compute the uniform density value (1 / volume)
        volume = (right_boundary - left_boundary) ** dim
        self.uniform_density = torch.tensor(1.0 / volume, device=self.device, dtype=torch.float32)
        self.log_uniform_density = torch.log(self.uniform_density)

    def get_name(self):
        """Return the name of the target distribution as a string."""
        return self.name
    
    def density(self, x):
        """
        Compute the probability density function (PDF) at point(s) x.
        
        Args:
            x: Tensor of shape (dim,) for single point or (batch_size, dim) for batch
            
        Returns:
            Tensor of densities with shape () for single point or (batch_size,) for batch
        """
        return torch.exp(self.log_density(x))
    
    def log_density(self, x):
        """
        Compute the log probability density function at point(s) x.
        
        Args:
            x: Tensor of shape (dim,) for single point or (batch_size, dim) for batch
            
        Returns:
            Tensor of log densities with shape () for single point or (batch_size,) for batch
        """
        if x.device != self.device:
            x = x.to(self.device)
        
        # Check if all coordinates are within boundaries
        if len(x.shape) == 1:
            # Single point: shape (dim,)
            within_bounds = torch.all((x >= self.left_boundary) & (x <= self.right_boundary))
            if within_bounds:
                return self.log_uniform_density
            else:
                return torch.tensor(-torch.inf, device=self.device, dtype=torch.float32)
        else:
            # Batch: shape (batch_size, dim)
            within_bounds = torch.all((x >= self.left_boundary) & (x <= self.right_boundary), dim=1)
            
            # Return log_uniform_density where within bounds, -inf otherwise
            log_densities = torch.where(
                within_bounds,
                self.log_uniform_density,
                torch.tensor(-torch.inf, device=self.device, dtype=torch.float32)
            )
            return log_densities

    def draw_sample(self, beta=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        # Convert to numpy for compatibility with existing code
        left_np = self.left_boundary.cpu().numpy()
        right_np = self.right_boundary.cpu().numpy()
        return np.random.uniform(left_np, right_np, self.dim)
    
    def draw_samples_torch(self, n_samples):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        Args:
            n_samples: Number of samples to draw
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Generate uniform samples on GPU
        samples = torch.rand(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        # Scale to the correct range
        samples = samples * (self.right_boundary - self.left_boundary) + self.left_boundary
        
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.left_boundary = self.left_boundary.to(device)
        self.right_boundary = self.right_boundary.to(device)
        self.uniform_density = self.uniform_density.to(device)
        self.log_uniform_density = self.log_uniform_density.to(device)
        return self 