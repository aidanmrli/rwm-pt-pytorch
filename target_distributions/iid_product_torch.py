import torch
import numpy as np
from interfaces.target_torch import TorchTargetDistribution

class IIDGammaTorch(TorchTargetDistribution):
    """
    PyTorch-native product of IID Gamma distributions with full GPU acceleration.
    No CPU-GPU transfers during density evaluation.
    """

    def __init__(self, dim, shape=2.0, scale=3.0, device=None):
        """
        Initialize the PyTorch-native IIDGamma distribution.

        Args:
            dim: Dimension of the distribution
            shape: Shape parameter of the Gamma distribution
            scale: Scale parameter of the Gamma distribution
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim, device)
        self.name = "IIDGammaTorch"
        
        # Store parameters as tensors on the device
        self.shape = torch.tensor(shape, device=self.device, dtype=torch.float32)
        self.scale = torch.tensor(scale, device=self.device, dtype=torch.float32)
        
        # Pre-compute log normalization constant for single Gamma distribution
        # log(Gamma(shape)) + shape * log(scale)
        self.log_gamma_shape = torch.lgamma(self.shape)
        self.log_norm_const_1d = self.log_gamma_shape + self.shape * torch.log(self.scale)
        
        # For d-dimensional product: d * log_norm_const_1d
        self.log_norm_const = dim * self.log_norm_const_1d

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
        
        # Check for valid domain (x > 0)
        if torch.any(x <= 0):
            if len(x.shape) == 1:
                return torch.tensor(-torch.inf, device=self.device, dtype=torch.float32)
            else:
                # Batch case: return -inf for invalid samples
                valid_mask = torch.all(x > 0, dim=1)
                log_densities = torch.full((x.shape[0],), -torch.inf, device=self.device, dtype=torch.float32)
                if torch.any(valid_mask):
                    x_valid = x[valid_mask]
                    log_densities[valid_mask] = self._compute_log_density_valid(x_valid)
                return log_densities
        
        return self._compute_log_density_valid(x)
    
    def _compute_log_density_valid(self, x):
        """Compute log density for valid inputs (x > 0)."""
        if len(x.shape) == 1:
            # Single point: shape (dim,)
            # log(x^(shape-1) * exp(-x/scale)) = (shape-1) * log(x) - x/scale
            log_density = torch.sum((self.shape - 1) * torch.log(x) - x / self.scale) - self.log_norm_const
            return log_density
        else:
            # Batch: shape (batch_size, dim)
            # Sum over dimensions for each sample
            log_densities = torch.sum((self.shape - 1) * torch.log(x) - x / self.scale, dim=1) - self.log_norm_const
            return log_densities

    def draw_sample(self, beta=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        # Convert to numpy for compatibility with existing code
        shape_np = self.shape.cpu().numpy()
        scale_np = self.scale.cpu().numpy()
        
        # Adjust shape parameter by beta (inverse temperature)
        adjusted_shape = shape_np * beta
        return np.random.gamma(adjusted_shape, scale_np, self.dim)
    
    def draw_samples_torch(self, n_samples, beta=1.0):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        Args:
            n_samples: Number of samples to draw
            beta: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Adjust shape parameter by beta
        adjusted_shape = self.shape * beta
        
        # PyTorch's Gamma distribution
        gamma_dist = torch.distributions.Gamma(adjusted_shape, 1.0 / self.scale)
        samples = gamma_dist.sample((n_samples, self.dim))
        
        return samples.to(self.device)

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.shape = self.shape.to(device)
        self.scale = self.scale.to(device)
        self.log_gamma_shape = self.log_gamma_shape.to(device)
        self.log_norm_const_1d = self.log_norm_const_1d.to(device)
        self.log_norm_const = self.log_norm_const.to(device)
        return self


class IIDBetaTorch(TorchTargetDistribution):
    """
    PyTorch-native product of IID Beta distributions with full GPU acceleration.
    No CPU-GPU transfers during density evaluation.
    
    The density of this target distribution is given by the product of d independent Beta distributions:
    f(x) = \prod_{i=1}^d \frac{1}{B(\alpha, \beta)} x_i^{\alpha-1} (1-x_i)^{\beta-1}
    where B(\alpha, \beta) is the Beta function.
    """

    def __init__(self, dim, alpha=2.0, beta=3.0, device=None):
        """
        Initialize the PyTorch-native IIDBeta distribution.

        Args:
            dim: Dimension of the distribution
            alpha: Alpha (shape1) parameter of the Beta distribution
            beta: Beta (shape2) parameter of the Beta distribution
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim, device)
        self.name = "IIDBetaTorch"
        
        # Store parameters as tensors on the device
        # Note: using beta to avoid confusion with temperature parameter beta
        self.alpha = torch.tensor(alpha, device=self.device, dtype=torch.float32)
        self.beta = torch.tensor(beta, device=self.device, dtype=torch.float32)
        
        # Pre-compute log normalization constant for single Beta distribution
        # log(Gamma(alpha + beta)) - log(Gamma(alpha)) - log(Gamma(beta))
        self.log_gamma_alpha = torch.lgamma(self.alpha)
        self.log_gamma_beta = torch.lgamma(self.beta)
        self.log_gamma_alpha_beta = torch.lgamma(self.alpha + self.beta)
        self.log_norm_const_1d = self.log_gamma_alpha_beta - self.log_gamma_alpha - self.log_gamma_beta
        
        # For d-dimensional product: d * log_norm_const_1d
        self.log_norm_const = dim * self.log_norm_const_1d

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
        
        # Check for valid domain (0 < x < 1)
        if torch.any((x <= 0) | (x >= 1)):
            if len(x.shape) == 1:
                return torch.tensor(-torch.inf, device=self.device, dtype=torch.float32)
            else:
                # Batch case: return -inf for invalid samples
                valid_mask = torch.all((x > 0) & (x < 1), dim=1)
                log_densities = torch.full((x.shape[0],), -torch.inf, device=self.device, dtype=torch.float32)
                if torch.any(valid_mask):
                    x_valid = x[valid_mask]
                    log_densities[valid_mask] = self._compute_log_density_valid(x_valid)
                return log_densities
        
        return self._compute_log_density_valid(x)
    
    def _compute_log_density_valid(self, x):
        """Compute log density for valid inputs (0 < x < 1)."""
        if len(x.shape) == 1:
            # Single point: shape (dim,)
            # log(x^(alpha-1) * (1-x)^(beta-1)) = (alpha-1) * log(x) + (beta-1) * log(1-x)
            log_density = torch.sum((self.alpha - 1) * torch.log(x) + 
                                  (self.beta - 1) * torch.log(1 - x)) + self.log_norm_const
            return log_density
        else:
            # Batch: shape (batch_size, dim)
            # Sum over dimensions for each sample
            log_densities = torch.sum((self.alpha - 1) * torch.log(x) + 
                                    (self.beta - 1) * torch.log(1 - x), dim=1) + self.log_norm_const
            return log_densities

    def draw_sample(self, beta_temp=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        # Convert to numpy for compatibility with existing code
        alpha_np = self.alpha.cpu().numpy()
        beta_np = self.beta.cpu().numpy()
        
        # Adjust parameters by beta (inverse temperature)
        adjusted_alpha = alpha_np * beta_temp
        adjusted_beta = beta_np * beta_temp
        
        return np.random.beta(adjusted_alpha, adjusted_beta, self.dim)
    
    def draw_samples_torch(self, n_samples, beta_temp=1.0):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        Args:
            n_samples: Number of samples to draw
            beta_temp: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Adjust parameters by beta
        adjusted_alpha = self.alpha * beta_temp
        adjusted_beta = self.beta * beta_temp
        
        # PyTorch's Beta distribution
        beta_dist = torch.distributions.Beta(adjusted_alpha, adjusted_beta)
        samples = beta_dist.sample((n_samples, self.dim))
        
        return samples.to(self.device)

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.alpha = self.alpha.to(device)
        self.beta = self.beta.to(device)
        self.log_gamma_alpha = self.log_gamma_alpha.to(device)
        self.log_gamma_beta = self.log_gamma_beta.to(device)
        self.log_gamma_alpha_beta = self.log_gamma_alpha_beta.to(device)
        self.log_norm_const_1d = self.log_norm_const_1d.to(device)
        self.log_norm_const = self.log_norm_const.to(device)
        return self 