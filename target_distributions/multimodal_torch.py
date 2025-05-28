import torch
import numpy as np
from interfaces.target_torch import TorchTargetDistribution

class ThreeMixtureDistributionTorch(TorchTargetDistribution):
    """
    PyTorch-native three-component mixture distribution with full GPU acceleration.
    Three modes at (-15, 0, ..., 0), (0, 0, ..., 0), and (15, 0, ..., 0).
    """

    def __init__(self, dim, scaling=False, device=None):
        """
        Initialize the PyTorch-native ThreeMixtureDistribution.

        Args:
            dim: Dimension of the distribution
            scaling: Whether to apply random scaling factors to coordinates
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim, device)
        self.name = "ThreeMixtureTorch"
        if scaling:
            self.name = "ThreeMixtureScaledTorch"
        
        # Initialize means: (-15, 0, ..., 0), (0, 0, ..., 0), (15, 0, ..., 0)
        self.means = torch.zeros(3, dim, device=self.device, dtype=torch.float32)
        self.means[0, 0] = -15.0
        self.means[2, 0] = 15.0
        
        # Initialize covariance matrices (identity matrices)
        self.covs = torch.eye(dim, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        
        if scaling:
            # Apply random scaling factors
            scaling_factors = torch.tensor(
                np.random.uniform(0.000001, 2, dim), 
                device=self.device, 
                dtype=torch.float32
            )
            self.scaling_factors = scaling_factors
            # Apply scaling to covariance matrices
            for i in range(3):
                self.covs[i] = self.covs[i] * scaling_factors.unsqueeze(0) * scaling_factors.unsqueeze(1)
        
        # Pre-compute inverse covariance matrices and determinants
        self.cov_invs = torch.linalg.inv(self.covs)
        self.cov_dets = torch.linalg.det(self.covs)
        
        # Pre-compute log normalization constants for each component
        log_2pi = torch.log(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32))
        self.log_norm_consts = -0.5 * (dim * log_2pi + torch.log(self.cov_dets))
        
        # Mixing weights (equal: 1/3 each)
        self.mixing_weights = torch.tensor([1/3, 1/3, 1/3], device=self.device, dtype=torch.float32)
        self.log_mixing_weights = torch.log(self.mixing_weights)

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
        
        if len(x.shape) == 1:
            # Single point: shape (dim,)
            log_component_densities = torch.zeros(3, device=self.device, dtype=torch.float32)
            
            for i in range(3):
                centered = x - self.means[i]
                quadratic_form = torch.dot(centered, torch.matmul(self.cov_invs[i], centered))
                log_component_densities[i] = -0.5 * quadratic_form + self.log_norm_consts[i] + self.log_mixing_weights[i]
            
            # Use logsumexp for numerical stability
            log_density = torch.logsumexp(log_component_densities, dim=0)
            return log_density
        else:
            # Batch: shape (batch_size, dim)
            batch_size = x.shape[0]
            log_component_densities = torch.zeros(batch_size, 3, device=self.device, dtype=torch.float32)
            
            for i in range(3):
                centered = x - self.means[i].unsqueeze(0)  # Broadcasting
                temp = torch.matmul(centered, self.cov_invs[i])
                quadratic_form = torch.sum(temp * centered, dim=1)
                log_component_densities[:, i] = (-0.5 * quadratic_form + 
                                               self.log_norm_consts[i] + 
                                               self.log_mixing_weights[i])
            
            # Use logsumexp for numerical stability
            log_densities = torch.logsumexp(log_component_densities, dim=1)
            return log_densities

    def draw_sample(self, beta=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        # Convert to numpy for compatibility with existing code
        means_np = self.means.cpu().numpy()
        covs_np = self.covs.cpu().numpy()
        
        # Pick a mode at random and sample from it
        random_integer = np.random.randint(0, 3)
        target_mean, target_cov = means_np[random_integer], covs_np[random_integer]
        
        return np.random.multivariate_normal(target_mean, target_cov / beta)
    
    def draw_samples_torch(self, n_samples, beta=1.0):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        Args:
            n_samples: Number of samples to draw
            beta: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        # Sample component indices
        component_indices = torch.multinomial(self.mixing_weights, n_samples, replacement=True)
        
        samples = torch.zeros(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        for i in range(3):
            mask = (component_indices == i)
            n_component_samples = mask.sum().item()
            
            if n_component_samples > 0:
                # Generate standard normal samples
                standard_samples = torch.randn(n_component_samples, self.dim, device=self.device, dtype=torch.float32)
                
                # Apply covariance structure
                cov_scaled = self.covs[i] / beta
                chol = torch.linalg.cholesky(cov_scaled)
                component_samples = self.means[i].unsqueeze(0) + torch.matmul(standard_samples, chol.T)
                
                samples[mask] = component_samples
        
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.means = self.means.to(device)
        self.covs = self.covs.to(device)
        self.cov_invs = self.cov_invs.to(device)
        self.cov_dets = self.cov_dets.to(device)
        self.log_norm_consts = self.log_norm_consts.to(device)
        self.mixing_weights = self.mixing_weights.to(device)
        self.log_mixing_weights = self.log_mixing_weights.to(device)
        if hasattr(self, 'scaling_factors'):
            self.scaling_factors = self.scaling_factors.to(device)
        return self


class RoughCarpetDistributionTorch(TorchTargetDistribution):
    """
    PyTorch-native rough carpet distribution with full GPU acceleration.
    Product of 1D three-mode distributions with modes at -15, 0, 15.
    """

    def __init__(self, dim, scaling=False, device=None):
        """
        Initialize the PyTorch-native RoughCarpetDistribution.

        Args:
            dim: Dimension of the distribution
            scaling: Whether to apply random scaling factors to coordinates
            device: PyTorch device for GPU acceleration
        """
        super().__init__(dim, device)
        self.name = "RoughCarpetTorch"
        if scaling:
            self.name = "RoughCarpetScaledTorch"
        
        # Modes and weights for 1D components
        self.modes = torch.tensor([-15.0, 0.0, 15.0], device=self.device, dtype=torch.float32)
        self.weights = torch.tensor([0.5, 0.3, 0.2], device=self.device, dtype=torch.float32)
        self.log_weights = torch.log(self.weights)
        
        # Pre-compute normalization constant for 1D Gaussian
        self.log_sqrt_2pi = torch.log(torch.sqrt(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32)))
        
        if scaling:
            # Apply random scaling factors
            scaling_factors = torch.tensor(
                np.random.uniform(0.000001, 2, dim), 
                device=self.device, 
                dtype=torch.float32
            )
            self.scaling_factors = scaling_factors

    def get_name(self):
        """Return the name of the target distribution as a string."""
        return self.name
    
    def density_1d(self, x):
        """
        Compute the density of a 1D three-mode distribution.
        
        Args:
            x: Tensor of shape () or (batch_size,)
            
        Returns:
            Tensor of densities with same shape as input
        """
        # Compute Gaussian densities for each mode
        # exp(-0.5 * (x - mode)^2) / sqrt(2*pi)
        x_expanded = x.unsqueeze(-1)  # Add mode dimension
        modes_expanded = self.modes.unsqueeze(0) if len(x.shape) > 0 else self.modes
        
        squared_diffs = (x_expanded - modes_expanded) ** 2
        log_densities = -0.5 * squared_diffs - self.log_sqrt_2pi + self.log_weights
        
        # Sum over modes using logsumexp for numerical stability
        log_density = torch.logsumexp(log_densities, dim=-1)
        return torch.exp(log_density)
    
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
        
        if hasattr(self, 'scaling_factors'):
            # Apply scaling transformation
            x_scaled = x * self.scaling_factors
            log_jacobian = torch.sum(torch.log(self.scaling_factors))
        else:
            x_scaled = x
            log_jacobian = 0.0
        
        if len(x.shape) == 1:
            # Single point: shape (dim,)
            log_density = 0.0
            for i in range(self.dim):
                # Compute log density for each coordinate
                x_i = x_scaled[i]
                squared_diffs = (x_i - self.modes) ** 2
                log_densities_i = -0.5 * squared_diffs - self.log_sqrt_2pi + self.log_weights
                log_density += torch.logsumexp(log_densities_i, dim=0)
            
            return log_density + log_jacobian
        else:
            # Batch: shape (batch_size, dim)
            batch_size = x.shape[0]
            log_densities = torch.zeros(batch_size, device=self.device, dtype=torch.float32)
            
            for i in range(self.dim):
                # Compute log density for each coordinate
                x_i = x_scaled[:, i]  # Shape: (batch_size,)
                x_i_expanded = x_i.unsqueeze(-1)  # Shape: (batch_size, 1)
                modes_expanded = self.modes.unsqueeze(0)  # Shape: (1, 3)
                
                squared_diffs = (x_i_expanded - modes_expanded) ** 2  # Shape: (batch_size, 3)
                log_densities_i = -0.5 * squared_diffs - self.log_sqrt_2pi + self.log_weights
                log_densities += torch.logsumexp(log_densities_i, dim=1)
            
            return log_densities + log_jacobian

    def draw_sample(self, beta=1.0):
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        sample = np.zeros(self.dim)
        modes_np = self.modes.cpu().numpy()
        weights_np = self.weights.cpu().numpy()
        
        for i in range(self.dim):
            mean_value = np.random.choice(modes_np, p=weights_np)
            sample[i] = np.random.normal(mean_value, 1.0 / np.sqrt(beta))
        
        return sample
    
    def draw_samples_torch(self, n_samples, beta=1.0):
        """
        Draw multiple samples using PyTorch (GPU-accelerated).
        
        Args:
            n_samples: Number of samples to draw
            beta: Inverse temperature parameter
            
        Returns:
            Tensor of samples with shape (n_samples, dim)
        """
        samples = torch.zeros(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        for i in range(self.dim):
            # Sample mode indices for this coordinate
            mode_indices = torch.multinomial(self.weights, n_samples, replacement=True)
            
            # Sample from normal distributions centered at selected modes
            selected_modes = self.modes[mode_indices]
            noise = torch.randn(n_samples, device=self.device, dtype=torch.float32) / torch.sqrt(torch.tensor(beta, device=self.device))
            samples[:, i] = selected_modes + noise
        
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.modes = self.modes.to(device)
        self.weights = self.weights.to(device)
        self.log_weights = self.log_weights.to(device)
        self.log_sqrt_2pi = self.log_sqrt_2pi.to(device)
        if hasattr(self, 'scaling_factors'):
            self.scaling_factors = self.scaling_factors.to(device)
        return self 