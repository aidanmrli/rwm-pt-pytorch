import torch
import numpy as np
from interfaces.target_torch import TorchTargetDistribution

class MultivariateNormalTorch(TorchTargetDistribution):
    """
    PyTorch-native multivariate normal distribution with full GPU acceleration.
    No CPU-GPU transfers during density evaluation.
    """

    def __init__(self, dim, mean=None, cov=None, device=None):
        """
        Initialize the PyTorch-native MultivariateNormal distribution.

        Args:
            dim: Dimension of the distribution
            mean: Mean vector of the distribution (defaults to zero)
            cov: Covariance matrix of the distribution (defaults to identity)
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim, device)
        self.name = "MultivariateNormalTorch"
        
        # Set default parameters
        if mean is None:
            mean = torch.zeros(dim, device=self.device, dtype=torch.float32)
        else:
            mean = torch.tensor(mean, device=self.device, dtype=torch.float32)
            
        if cov is None:
            cov = torch.eye(dim, device=self.device, dtype=torch.float32)
        else:
            cov = torch.tensor(cov, device=self.device, dtype=torch.float32)
        
        self.mean = mean
        self.cov = cov
        
        # Pre-compute inverse and determinant for efficiency
        self.cov_inv = torch.linalg.inv(self.cov)
        self.cov_det = torch.linalg.det(self.cov)
        
        # Pre-compute log normalization constant
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32))
        self.log_norm_const = -0.5 * (dim * log_2pi + torch.log(self.cov_det))

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
        
        # Handle both single point and batch evaluation
        if len(x.shape) == 1:
            # Single point: shape (dim,)
            centered = x - self.mean
            quadratic_form = torch.dot(centered, torch.matmul(self.cov_inv, centered))
            log_density = -0.5 * quadratic_form + self.log_norm_const
            return log_density
        else:
            # Batch: shape (batch_size, dim)
            centered = x - self.mean.unsqueeze(0)  # Broadcasting
            
            # Compute quadratic form: (x - mu)^T Σ^(-1) (x - mu)
            temp = torch.matmul(centered, self.cov_inv)
            quadratic_form = torch.sum(temp * centered, dim=1)
            
            # Compute log densities
            log_densities = -0.5 * quadratic_form + self.log_norm_const
            return log_densities

    def draw_sample(self, beta=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        # Convert to numpy for compatibility with existing code
        mean_np = self.mean.cpu().numpy()
        cov_np = self.cov.cpu().numpy()
        return np.random.multivariate_normal(mean_np, cov_np / beta)
    
    def draw_samples_torch(self, n_samples, beta=1.0):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        Args:
            n_samples: Number of samples to draw
            beta: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Generate standard normal samples on GPU
        standard_samples = torch.randn(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        # Apply covariance structure: samples = mean + chol(cov/beta) @ noise^T
        cov_scaled = self.cov / beta
        chol = torch.linalg.cholesky(cov_scaled)
        
        samples = self.mean.unsqueeze(0) + torch.matmul(standard_samples, chol.T)
        
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.mean = self.mean.to(device)
        self.cov = self.cov.to(device)
        self.cov_inv = self.cov_inv.to(device)
        self.cov_det = self.cov_det.to(device)
        self.log_norm_const = self.log_norm_const.to(device)
        return self 