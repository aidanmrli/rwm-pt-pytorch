import torch
import numpy as np
from interfaces.target_torch import TorchTargetDistribution

class ScaledMultivariateNormalTorch(TorchTargetDistribution):
    """
    PyTorch-native scaled multivariate normal distribution with independent component densities.
    Each dimension has its own variance, creating the target density as a product of iid components:
    π(x) = Π f_i(x_i) where f_i(x_i) = N(x_i | μ_i, C_i)
    
    This can also be written as π(x) = Π c_i N(c_i x_i | 0, 1) where c_i = 1/√C_i.
    """

    def __init__(self, dim, mean=None, variances=None, scaling_factors=None, 
                 variance_range=(0.02, 1.98), device=None, seed=None):
        """
        Initialize the scaled multivariate normal distribution.

        Args:
            dim: Dimension of the distribution
            mean: Mean vector of the distribution (defaults to zero)
            variances: Tensor of variances C_i for each dimension (optional)
            scaling_factors: Tensor of scaling factors c_i = 1/√C_i (optional)
            variance_range: Tuple (min_var, max_var) for sampling variances if not provided
            device: PyTorch device for GPU acceleration
            seed: Random seed for reproducible variance sampling
        """
        super().__init__(dim, device)
        self.name = "ScaledMultivariateNormalTorch"
        
        if seed is not None:
            torch.manual_seed(seed)
        
        # Set default mean
        if mean is None:
            mean = torch.zeros(dim, device=self.device, dtype=torch.float32)
        else:
            if isinstance(mean, torch.Tensor):
                mean = mean.clone().detach().to(device=self.device, dtype=torch.float32)
            else:
                mean = torch.tensor(mean, device=self.device, dtype=torch.float32)
        
        self.mean = mean
        
        # Handle variance/scaling factor specification
        if variances is not None and scaling_factors is not None:
            raise ValueError("Cannot specify both variances and scaling_factors")
        
        if variances is not None:
            # Direct variance specification
            if isinstance(variances, torch.Tensor):
                self.variances = variances.clone().detach().to(device=self.device, dtype=torch.float32)
            else:
                self.variances = torch.tensor(variances, device=self.device, dtype=torch.float32)
            self.scaling_factors = 1.0 / torch.sqrt(self.variances)
        elif scaling_factors is not None:
            # Direct scaling factor specification
            if isinstance(scaling_factors, torch.Tensor):
                self.scaling_factors = scaling_factors.clone().detach().to(device=self.device, dtype=torch.float32)
            else:
                self.scaling_factors = torch.tensor(scaling_factors, device=self.device, dtype=torch.float32)
            self.variances = 1.0 / (self.scaling_factors ** 2)
        else:
            # Sample variances from uniform distribution
            min_var, max_var = variance_range
            self.variances = torch.rand(dim, device=self.device, dtype=torch.float32) * (max_var - min_var) + min_var
            self.scaling_factors = 1.0 / torch.sqrt(self.variances)
        
        # Verify dimensions
        assert self.variances.shape == (dim,), f"Variances must have shape ({dim},), got {self.variances.shape}"
        assert self.scaling_factors.shape == (dim,), f"Scaling factors must have shape ({dim},), got {self.scaling_factors.shape}"
        
        # Pre-compute log normalization constant
        # For independent Gaussians: log(Z) = Σ log(√(2π C_i)) = 0.5 * Σ log(2π C_i)
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32))
        self.log_norm_const = -0.5 * torch.sum(log_2pi + torch.log(self.variances))

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
        π(x) = Π N(x_i | μ_i, C_i) = Π (1/√(2π C_i)) exp(-0.5 (x_i - μ_i)²/C_i)
        log π(x) = Σ (-0.5 log(2π C_i) - 0.5 (x_i - μ_i)²/C_i)
        
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
            # Sum of independent log densities: -0.5 * Σ (x_i - μ_i)²/C_i
            scaled_squared_diff = (centered ** 2) / self.variances
            log_density = -0.5 * torch.sum(scaled_squared_diff) + self.log_norm_const
            return log_density
        else:
            # Batch: shape (batch_size, dim)
            centered = x - self.mean.unsqueeze(0)  # Broadcasting
            
            # Sum of independent log densities for each sample in batch
            scaled_squared_diff = (centered ** 2) / self.variances.unsqueeze(0)
            log_densities = -0.5 * torch.sum(scaled_squared_diff, dim=1) + self.log_norm_const
            return log_densities

    def log_density_rr_form(self, x):
        """
        Compute log density using R&R (2001) form: π(x) = Π c_i N(c_i x_i | 0, 1).
        This is mathematically equivalent to the standard form but computed differently.
        
        Args:
            x: Tensor of shape (dim,) for single point or (batch_size, dim) for batch
            
        Returns:
            Tensor of log densities with shape () for single point or (batch_size,) for batch
        """
        if x.device != self.device:
            x = x.to(self.device)
        
        # R&R form: log π(x) = Σ (log c_i - 0.5 log(2π) - 0.5 (c_i x_i)²)
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32))
        
        if len(x.shape) == 1:
            # Single point
            scaled_x = self.scaling_factors * x
            log_density = (torch.sum(torch.log(self.scaling_factors)) - 
                          0.5 * self.dim * log_2pi - 
                          0.5 * torch.sum(scaled_x ** 2))
            return log_density
        else:
            # Batch
            scaled_x = self.scaling_factors.unsqueeze(0) * x
            log_densities = (torch.sum(torch.log(self.scaling_factors)) - 
                           0.5 * self.dim * log_2pi - 
                           0.5 * torch.sum(scaled_x ** 2, dim=1))
            return log_densities

    def draw_sample(self, beta=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        # Convert to numpy for compatibility with existing code
        mean_np = self.mean.cpu().numpy()
        variances_np = self.variances.cpu().numpy()
        
        # Sample each dimension independently
        samples = np.zeros(self.dim)
        for i in range(self.dim):
            samples[i] = np.random.normal(mean_np[i], np.sqrt(variances_np[i] / beta))
        
        return samples
    
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
        
        # Scale by standard deviations: samples = mean + sqrt(variances/beta) * noise
        std_devs = torch.sqrt(self.variances / beta)
        samples = self.mean.unsqueeze(0) + std_devs.unsqueeze(0) * standard_samples
        
        return samples

    def get_variances(self):
        """Return the variances C_i for each dimension."""
        return self.variances.clone()
    
    def get_scaling_factors(self):
        """Return the scaling factors c_i = 1/√C_i for each dimension."""
        return self.scaling_factors.clone()
    
    def get_diagonal_covariance_matrix(self):
        """Return the diagonal covariance matrix with variances on the diagonal."""
        return torch.diag(self.variances)

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.mean = self.mean.to(device)
        self.variances = self.variances.to(device)
        self.scaling_factors = self.scaling_factors.to(device)
        self.log_norm_const = self.log_norm_const.to(device)
        return self

    def __repr__(self):
        """String representation of the distribution."""
        return (f"ScaledMultivariateNormalTorch(dim={self.dim}, "
                f"variance_range=({self.variances.min().item():.4f}, {self.variances.max().item():.4f}), "
                f"device={self.device})") 