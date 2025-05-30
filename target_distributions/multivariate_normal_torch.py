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


class ScaledMultivariateNormalTorch(TorchTargetDistribution):
    """
    PyTorch-native scaled multivariate normal distribution implementing:
    π(x) = Π c_i N(c_i * x_i | 0, 1)
    
    Where:
    - c_i are scaling factors
    - N(y | 0, 1) is the standard normal density
    - This creates a product of independent scaled standard normal components
    """

    def __init__(self, dim, scaling_factors=None, scaling_range=(0.02, 1.98), device=None, seed=None):
        """
        Initialize the scaled multivariate normal distribution.

        Args:
            dim: Dimension of the distribution
            scaling_factors: Tensor of scaling factors c_i (optional)
            scaling_range: Tuple (min_scale, max_scale) for sampling scaling factors if not provided
            device: PyTorch device for GPU acceleration
            seed: Random seed for reproducible scaling factor sampling
        """
        super().__init__(dim, device)
        self.name = "ScaledMultivariateNormalTorch"
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Handle scaling factor specification
        if scaling_factors is not None:
            # Direct scaling factor specification
            if isinstance(scaling_factors, torch.Tensor):
                self.scaling_factors = scaling_factors.clone().detach().to(device=self.device, dtype=torch.float32)
            else:
                self.scaling_factors = torch.tensor(scaling_factors, device=self.device, dtype=torch.float32)
        else:
            # Sample scaling factors from uniform distribution
            min_scale, max_scale = scaling_range
            self.scaling_factors = torch.rand(dim, device=self.device, dtype=torch.float32) * (max_scale - min_scale) + min_scale
        
        # Verify dimensions
        assert self.scaling_factors.shape == (dim,), f"Scaling factors must have shape ({dim},), got {self.scaling_factors.shape}"
        
        # Pre-compute log normalization constant for efficiency
        # log π(x) = Σ log(c_i) - 0.5 * D * log(2π) - 0.5 * Σ (c_i * x_i)²
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32))
        self.log_norm_const = torch.sum(torch.log(self.scaling_factors)) - 0.5 * self.dim * log_2pi

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
        π(x) = Π c_i N(c_i * x_i | 0, 1)
        log π(x) = Σ log(c_i) - 0.5 * D * log(2π) - 0.5 * Σ (c_i * x_i)²
        
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
            scaled_x = self.scaling_factors * x
            log_density = self.log_norm_const - 0.5 * torch.sum(scaled_x ** 2)
            return log_density
        else:
            # Batch: shape (batch_size, dim)
            scaled_x = self.scaling_factors.unsqueeze(0) * x  # Broadcasting
            log_densities = self.log_norm_const - 0.5 * torch.sum(scaled_x ** 2, dim=1)
            return log_densities

    def draw_sample(self, beta=1.0):
        """
        Draw a sample from the distribution (CPU implementation for compatibility).
        
        For π(x) = Π c_i N(c_i * x_i | 0, 1), we have:
        c_i * x_i ~ N(0, 1), so x_i ~ N(0, 1/c_i²)
        
        Args:
            beta: Inverse temperature parameter
            
        Returns:
            numpy array sample of shape (dim,)
        """
        # Convert to numpy for compatibility with existing code
        scaling_factors_np = self.scaling_factors.cpu().numpy()
        
        # Sample each dimension independently: x_i ~ N(0, 1/(c_i²*beta))
        samples = np.zeros(self.dim)
        for i in range(self.dim):
            std_dev = 1.0 / (scaling_factors_np[i] * np.sqrt(beta))
            samples[i] = np.random.normal(0.0, std_dev)
        
        return samples
    
    def draw_samples_torch(self, n_samples, beta=1.0):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        For π(x) = Π c_i N(c_i * x_i | 0, 1), we have:
        c_i * x_i ~ N(0, 1), so x_i ~ N(0, 1/c_i²)
        
        Args:
            n_samples: Number of samples to draw
            beta: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Generate standard normal samples on GPU
        standard_samples = torch.randn(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        # Scale by standard deviations: x_i ~ N(0, 1/(c_i²*beta))
        # std_dev_i = 1/(c_i * sqrt(beta))
        std_devs = 1.0 / (self.scaling_factors * torch.sqrt(torch.tensor(beta, device=self.device, dtype=torch.float32)))
        samples = std_devs.unsqueeze(0) * standard_samples
        
        return samples

    def get_scaling_factors(self):
        """Return the scaling factors c_i."""
        return self.scaling_factors.clone()
    
    def get_variances(self):
        """Return the equivalent variances 1/c_i² for each dimension."""
        return 1.0 / (self.scaling_factors ** 2)
    
    def get_diagonal_covariance_matrix(self):
        """Return the diagonal covariance matrix with equivalent variances 1/c_i² on the diagonal."""
        return torch.diag(self.get_variances())

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.scaling_factors = self.scaling_factors.to(device)
        self.log_norm_const = self.log_norm_const.to(device)
        return self

    def __repr__(self):
        """String representation of the distribution."""
        return (f"ScaledMultivariateNormalTorch(dim={self.dim}, "
                f"scaling_range=({self.scaling_factors.min().item():.4f}, {self.scaling_factors.max().item():.4f}), "
                f"device={self.device})") 