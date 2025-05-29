import torch
from interfaces.target_torch import TorchTargetDistribution

class ThreeMixtureDistributionTorch(TorchTargetDistribution):
    """
    PyTorch-native three-component mixture distribution with full GPU acceleration.
    Customizable mode centers and weights for flexible multimodal distributions.
    """

    def __init__(self, dim, scaling=False, device=None, mode_centers=None, mode_weights=None):
        """
        Initialize the PyTorch-native ThreeMixtureDistribution.

        Args:
            dim: Dimension of the distribution
            scaling: Whether to apply random scaling factors to coordinates
            device: PyTorch device for GPU acceleration
            mode_centers: List/array of 3 mode centers, each of length dim.
                         Default: [[-5, 0, ..., 0], [0, 0, ..., 0], [5, 0, ..., 0]]
            mode_weights: List/array of 3 mixing weights that sum to 1.
                         Default: [1/3, 1/3, 1/3]
        """
        super().__init__(dim, device)
        
        # Set default mode centers if not provided
        if mode_centers is None:
            mode_centers = [
                [-5.0] + [0.0] * (dim - 1),  # (-5, 0, ..., 0)
                [0.0] * dim,                   # (0, 0, ..., 0)
                [5.0] + [0.0] * (dim - 1)     # (5, 0, ..., 0)
            ]
        
        # Set default mode weights if not provided
        if mode_weights is None:
            mode_weights = [1/3, 1/3, 1/3]
        
        # Validate inputs
        self._validate_mode_parameters(mode_centers, mode_weights, dim)
        
        # Generate name based on parameters
        self.name = self._generate_name(scaling, mode_centers, mode_weights)
        
        # Initialize means from mode_centers
        self.means = torch.tensor(mode_centers, device=self.device, dtype=torch.float32)
        
        # Initialize covariance matrices (identity matrices)
        self.covs = torch.eye(dim, device=self.device, dtype=torch.float32).unsqueeze(0).repeat(3, 1, 1)
        
        if scaling:
            # Apply random scaling factors uniformly from [0.02, 1.98] with expectation 1.0
            scaling_factors = torch.rand(dim, device=self.device, dtype=torch.float32) * (1.98 - 0.02) + 0.02
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
        
        # Initialize mixing weights
        self.mixing_weights = torch.tensor(mode_weights, device=self.device, dtype=torch.float32)
        self.log_mixing_weights = torch.log(self.mixing_weights)

    def _validate_mode_parameters(self, mode_centers, mode_weights, dim):
        """Validate mode centers and weights parameters."""
        # Validate mode_centers
        if len(mode_centers) != 3:
            raise ValueError(f"mode_centers must contain exactly 3 modes, got {len(mode_centers)}")
        
        for i, center in enumerate(mode_centers):
            if len(center) != dim:
                raise ValueError(f"Mode {i} has dimension {len(center)}, expected {dim}")
        
        # Validate mode_weights
        if len(mode_weights) != 3:
            raise ValueError(f"mode_weights must contain exactly 3 weights, got {len(mode_weights)}")
        
        weights_tensor = torch.tensor(mode_weights, dtype=torch.float32)
        if not torch.all(weights_tensor > 0):
            raise ValueError("All mode_weights must be positive")
        
        if not torch.allclose(torch.sum(weights_tensor), torch.tensor(1.0), rtol=1e-6):
            raise ValueError(f"mode_weights must sum to 1.0, got sum = {torch.sum(weights_tensor).item()}")

    def _generate_name(self, scaling, mode_centers, mode_weights):
        """Generate a descriptive name based on parameters."""
        base_name = "ThreeMixtureTorch"
        
        # Check if using default parameters
        default_centers = [
            [-5.0] + [0.0] * (self.dim - 1),
            [0.0] * self.dim,
            [5.0] + [0.0] * (self.dim - 1)
        ]
        default_weights = [1/3, 1/3, 1/3]
        
        # Convert to tensors for comparison
        mode_centers_tensor = torch.tensor(mode_centers, dtype=torch.float32)
        default_centers_tensor = torch.tensor(default_centers, dtype=torch.float32)
        mode_weights_tensor = torch.tensor(mode_weights, dtype=torch.float32)
        default_weights_tensor = torch.tensor(default_weights, dtype=torch.float32)
        
        is_default_centers = torch.allclose(mode_centers_tensor, default_centers_tensor, rtol=1e-6)
        is_default_weights = torch.allclose(mode_weights_tensor, default_weights_tensor, rtol=1e-6)
        
        if not (is_default_centers and is_default_weights):
            base_name += "Custom"
        
        if scaling:
            base_name += "Scaled"
        
        return base_name

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
        # Use PyTorch for sampling but return numpy for compatibility
        with torch.no_grad():
            # Pick a mode at random and sample from it
            random_integer = torch.randint(0, 3, (1,), device=self.device).item()
            target_mean, target_cov = self.means[random_integer], self.covs[random_integer]
            
            # Generate sample using PyTorch multivariate normal
            cov_scaled = target_cov / beta
            chol = torch.linalg.cholesky(cov_scaled)
            standard_sample = torch.randn(self.dim, device=self.device, dtype=torch.float32)
            sample = target_mean + torch.matmul(chol, standard_sample)
            
            return sample.cpu().numpy()
    
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
    Product of 1D three-mode distributions with customizable mode centers and weights.
    """

    def __init__(self, dim, scaling=False, device=None, mode_centers=None, mode_weights=None):
        """
        Initialize the PyTorch-native RoughCarpetDistribution.

        Args:
            dim: Dimension of the distribution
            scaling: Whether to apply random scaling factors to coordinates
            device: PyTorch device for GPU acceleration
            mode_centers: List/array of 3 scalar mode centers for 1D components.
                         Default: [-5.0, 0.0, 5.0]
            mode_weights: List/array of 3 mixing weights that sum to 1.
                         Default: [0.5, 0.3, 0.2]
        """
        super().__init__(dim, device)
        
        # Set default mode centers if not provided
        if mode_centers is None:
            mode_centers = [-5.0, 0.0, 5.0]
        
        # Set default mode weights if not provided
        if mode_weights is None:
            mode_weights = [0.5, 0.3, 0.2]
        
        # Validate inputs
        self._validate_mode_parameters(mode_centers, mode_weights)
        
        # Generate name based on parameters
        self.name = self._generate_name(scaling, mode_centers, mode_weights)
        
        # Modes and weights for 1D components
        self.modes = torch.tensor(mode_centers, device=self.device, dtype=torch.float32)
        self.weights = torch.tensor(mode_weights, device=self.device, dtype=torch.float32)
        self.log_weights = torch.log(self.weights)
        
        # Pre-compute normalization constant for 1D Gaussian
        self.log_sqrt_2pi = torch.log(torch.sqrt(torch.tensor(2.0 * torch.pi, device=self.device, dtype=torch.float32)))
        
        if scaling:
            # Apply random scaling factors uniformly from [0.02, 1.98] with expectation 1.0
            scaling_factors = torch.rand(dim, device=self.device, dtype=torch.float32) * (1.98 - 0.02) + 0.02
            self.scaling_factors = scaling_factors

    def _validate_mode_parameters(self, mode_centers, mode_weights):
        """Validate mode centers and weights parameters."""
        # Validate mode_centers
        if len(mode_centers) != 3:
            raise ValueError(f"mode_centers must contain exactly 3 modes, got {len(mode_centers)}")
        
        # Check that mode_centers are scalars (not arrays)
        for i, center in enumerate(mode_centers):
            if not isinstance(center, (int, float)):
                raise ValueError(f"Mode center {i} must be a scalar, got {type(center)}")
        
        # Validate mode_weights
        if len(mode_weights) != 3:
            raise ValueError(f"mode_weights must contain exactly 3 weights, got {len(mode_weights)}")
        
        weights_tensor = torch.tensor(mode_weights, dtype=torch.float32)
        if not torch.all(weights_tensor > 0):
            raise ValueError("All mode_weights must be positive")
        
        if not torch.allclose(torch.sum(weights_tensor), torch.tensor(1.0), rtol=1e-6):
            raise ValueError(f"mode_weights must sum to 1.0, got sum = {torch.sum(weights_tensor).item()}")

    def _generate_name(self, scaling, mode_centers, mode_weights):
        """Generate a descriptive name based on parameters."""
        base_name = "RoughCarpetTorch"
        
        # Check if using default parameters
        default_centers = [-15.0, 0.0, 15.0]
        default_weights = [0.5, 0.3, 0.2]
        
        # Convert to tensors for comparison
        mode_centers_tensor = torch.tensor(mode_centers, dtype=torch.float32)
        default_centers_tensor = torch.tensor(default_centers, dtype=torch.float32)
        mode_weights_tensor = torch.tensor(mode_weights, dtype=torch.float32)
        default_weights_tensor = torch.tensor(default_weights, dtype=torch.float32)
        
        is_default_centers = torch.allclose(mode_centers_tensor, default_centers_tensor, rtol=1e-6)
        is_default_weights = torch.allclose(mode_weights_tensor, default_weights_tensor, rtol=1e-6)
        
        if not (is_default_centers and is_default_weights):
            base_name += "Custom"
        
        if scaling:
            base_name += "Scaled"
        
        return base_name

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
        # Use PyTorch for sampling but return numpy for compatibility
        with torch.no_grad():
            sample = torch.zeros(self.dim, device=self.device, dtype=torch.float32)
            
            for i in range(self.dim):
                # Sample mode index for this coordinate
                mode_idx = torch.multinomial(self.weights, 1, replacement=True).item()
                mean_value = self.modes[mode_idx]
                
                # Sample from normal distribution
                noise = torch.randn(1, device=self.device, dtype=torch.float32) / torch.sqrt(torch.tensor(beta, device=self.device))
                sample[i] = mean_value + noise.item()
            
            return sample.cpu().numpy()
    
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