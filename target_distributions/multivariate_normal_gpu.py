import numpy as np
import torch
from interfaces import TargetDistribution
from scipy.stats import norm, multivariate_normal

class MultivariateNormal_GPU(TargetDistribution):
    """
    GPU-accelerated multivariate normal distribution with batch density evaluation.
    Provides significant speedup for large batch sizes and high dimensions.
    """

    def __init__(self, dim, mean=None, cov=None, device=None):
        """
        Initialize the GPU-accelerated MultivariateNormal distribution.

        Args:
            dim: Dimension of the distribution
            mean: Mean vector of the distribution (defaults to zero)
            cov: Covariance matrix of the distribution (defaults to identity)
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim)
        self.name = "MultivariateNormal_GPU"
        
        # Setup device
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Set default parameters
        if mean is None:
            mean = np.zeros(dim)
        if cov is None:
            cov = np.eye(dim)
        
        self.mean = mean
        self.cov = cov
        
        # Convert to GPU tensors
        self.mean_gpu = torch.tensor(mean, device=self.device, dtype=torch.float32)
        self.cov_gpu = torch.tensor(cov, device=self.device, dtype=torch.float32)
        
        # Pre-compute inverse and determinant for efficiency
        self.cov_inv_gpu = torch.linalg.inv(self.cov_gpu)
        self.cov_det = torch.linalg.det(self.cov_gpu).item()
        
        # Pre-compute normalization constant
        self.norm_const = 1.0 / np.sqrt((2 * np.pi) ** dim * self.cov_det)
        self.log_norm_const = np.log(self.norm_const)

    def get_name(self):
        """Return the name of the target distribution as a string."""
        return self.name
    
    def density_1d(self, x):
        """
        Evaluate 1D density (for compatibility with base class).
        """
        return norm.pdf(x, loc=self.mean[0], scale=np.sqrt(self.cov[0][0]))
    
    def density(self, x):
        """
        Evaluate the probability density function (PDF) at a point x.
        Falls back to CPU implementation for single point evaluation.
        """
        if isinstance(x, (int, float)) or len(x) == 1:
            return self.density_1d(x)
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)
    
    def batch_density_gpu(self, x_batch):
        """
        GPU-accelerated batch evaluation of the PDF.
        
        Args:
            x_batch: Tensor of shape (batch_size, dim) or (dim,) for single point
            
        Returns:
            Tensor of densities with shape (batch_size,) or scalar
        """
        if len(x_batch.shape) == 1:
            x_batch = x_batch.unsqueeze(0)
        
        # Center the data: x - mu
        centered = x_batch - self.mean_gpu.unsqueeze(0)
        
        # Compute quadratic form: (x - mu)^T Σ^(-1) (x - mu)
        # Shape: (batch_size, dim) @ (dim, dim) = (batch_size, dim)
        temp = torch.matmul(centered, self.cov_inv_gpu)
        # Shape: (batch_size, dim) * (batch_size, dim) -> (batch_size,)
        quadratic_form = torch.sum(temp * centered, dim=1)
        
        # Compute densities: exp(-0.5 * quadratic_form) * norm_const
        log_densities = -0.5 * quadratic_form + self.log_norm_const
        densities = torch.exp(log_densities)
        
        return densities
    
    def log_density_gpu(self, x_batch):
        """
        GPU-accelerated batch evaluation of log PDF (more numerically stable).
        
        Args:
            x_batch: Tensor of shape (batch_size, dim) or (dim,) for single point
            
        Returns:
            Tensor of log densities with shape (batch_size,) or scalar
        """
        if len(x_batch.shape) == 1:
            x_batch = x_batch.unsqueeze(0)
        
        # Center the data
        centered = x_batch - self.mean_gpu.unsqueeze(0)
        
        # Compute quadratic form
        temp = torch.matmul(centered, self.cov_inv_gpu)
        quadratic_form = torch.sum(temp * centered, dim=1)
        
        # Compute log densities
        log_densities = -0.5 * quadratic_form + self.log_norm_const
        
        return log_densities

    def draw_sample(self, beta=1):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        return np.random.multivariate_normal(self.mean, self.cov / beta)
    
    def draw_samples_gpu(self, n_samples, beta=1):
        """
        Draw multiple samples using GPU acceleration.
        
        Args:
            n_samples: Number of samples to draw
            beta: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Generate standard normal samples on GPU
        standard_samples = torch.randn(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        # Apply covariance structure: samples = mean + chol(cov/beta) @ noise^T
        cov_scaled = self.cov_gpu / beta
        chol = torch.linalg.cholesky(cov_scaled)
        
        samples = self.mean_gpu.unsqueeze(0) + torch.matmul(standard_samples, chol.T)
        
        return samples 