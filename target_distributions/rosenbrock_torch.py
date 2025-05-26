import torch
import numpy as np
import math
from typing import Union
from interfaces.target_torch import TorchTargetDistribution

# Default Rosenbrock parameters from the paper
DEFAULT_A_COEFF = 1.0 / 20.0
DEFAULT_B_COEFF = 100.0 / 20.0
DEFAULT_MU = 1.0


class FullRosenbrockTorch(TorchTargetDistribution):
    """
    N-dimensional Full Rosenbrock distribution as defined in eq. 2 of
    "An n-dimensional Rosenbrock Distribution for MCMC testing" by Pagani et al. (2022).
    
    log_prob(x) = - sum_{i=1}^{n-1} [b_coeff * (x_{i+1} - x_i^2)^2 + a_coeff * (x_i - mu_i)^2]
    where x_i is the i-th component of x (0-indexed internally) and mu_i is the i-th component of mu.
    Note that mu_i = 1 for all i by default to match the original paper.
    Note that the original paper uses a_coeff = 1/20 and b_coeff = 100/20 by default.
    """
    
    def __init__(self, dim: int, a_coeff: float = DEFAULT_A_COEFF, b_coeff: float = DEFAULT_B_COEFF, 
                 mu: Union[float, torch.Tensor] = DEFAULT_MU, device: str = None):
        if dim < 2:
            raise ValueError("Dimension for FullRosenbrockTorch must be at least 2.")
        super().__init__(dim, device)
        
        self.a_coeff = torch.tensor(a_coeff, device=self.device, dtype=torch.float32)
        self.b_coeff = torch.tensor(b_coeff, device=self.device, dtype=torch.float32)
        
        # mu can be a scalar or a tensor of shape (dim-1) for x_i, i=0 to dim-2
        if isinstance(mu, (int, float)):
            self.mu = torch.full((dim - 1,), mu, device=self.device, dtype=torch.float32)
        elif isinstance(mu, torch.Tensor):
            if mu.ndim == 0:  # scalar tensor
                self.mu = torch.full((dim - 1,), mu.item(), device=self.device, dtype=torch.float32)
            elif mu.shape == (dim - 1,):
                self.mu = mu.to(device=self.device, dtype=torch.float32)
            else:
                raise ValueError(f"mu tensor must be scalar or have shape ({dim-1},)")
        else:
            raise TypeError("mu must be float, int, or torch.Tensor")
        
        self._name = f"FullRosenbrockTorch"

    def _validate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and reshape input tensor to ensure it is (batch_size, dim)."""
        if x.device != self.device:
            x = x.to(self.device)
        
        if x.ndim == 1:
            # Single point: reshape to (1, dim)
            if x.shape[0] != self.dim:
                raise ValueError(f"Expected tensor of shape ({self.dim},), got {x.shape}")
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            # Batch: ensure correct dimension
            if x.shape[1] != self.dim:
                raise ValueError(f"Expected tensor of shape (batch_size, {self.dim}), got {x.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {x.ndim}D tensor")
        
        return x

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_tensor(x)  # Ensure x is (batch_size, dim)
        
        # x_i terms are x[:, :-1] (all but the last column for each sample)
        # x_{i+1} terms are x[:, 1:] (all but the first column for each sample)
        x_i = x[:, :-1]
        x_i_plus_1 = x[:, 1:]
        
        term1 = self.b_coeff * (x_i_plus_1 - x_i**2)**2
        term2 = self.a_coeff * (x_i - self.mu)**2  # mu is broadcasted if necessary
        # Defaults: a_coeff = 1/20, b_coeff = 100/20, mu_i = 1.0 for all i.
        
        log_prob = -(torch.sum(term1, dim=1) + torch.sum(term2, dim=1))
        
        # Return scalar for single point, tensor for batch
        if log_prob.shape[0] == 1:
            return log_prob.squeeze(0)
        return log_prob

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the density of the distribution at point(s) x."""
        return torch.exp(self.log_density(x))

    def get_name(self) -> str:
        return self._name

    def draw_sample(self, beta: float = 1.0) -> np.ndarray:
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        samples = self.draw_samples_torch(1, beta)
        return samples[0].cpu().numpy()

    def draw_samples_torch(self, n_samples: int, beta: float = 1.0) -> torch.Tensor:
        """
        Draw multiple samples using approximate conditional sampling for initialization.
        This is a heuristic as per TorchTargetDistribution's draw_sample purpose.
        """
        raise NotImplementedError("Draw samples for FullRosenbrockTorch is not implemented yet.")
        samples = torch.zeros(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        # Effective a and b for sampling, scaled by beta (inverse temperature)
        eff_a = self.a_coeff * beta
        eff_b = self.b_coeff * beta

        # Sample x_0 (first component of x_i)
        # x_0 ~ N(mu_0, 1/(2*eff_a))
        mu0 = self.mu[0] if self.mu.ndim > 0 else self.mu 
        var_0 = 1.0 / (2 * eff_a) if eff_a > 0 else 1.0
        samples[:, 0] = mu0 + torch.randn(n_samples, device=self.device, dtype=torch.float32) * torch.sqrt(var_0)
        
        for i in range(self.dim - 1):
            # x_{i+1} | x_i ~ N(x_i^2, 1/(2*eff_b))
            x_i_sq = samples[:, i]**2
            var_i = 1.0 / (2 * eff_b) if eff_b > 0 else 1.0
            samples[:, i+1] = x_i_sq + torch.randn(n_samples, device=self.device, dtype=torch.float32) * torch.sqrt(var_i)
        
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.a_coeff = self.a_coeff.to(device)
        self.b_coeff = self.b_coeff.to(device)
        self.mu = self.mu.to(device)
        return self


class EvenRosenbrockTorch(TorchTargetDistribution):
    """
    N-dimensional Even Rosenbrock distribution as defined in eq. 3 of Pagani et al. (2022).
    Dimension n must be even.
    
    log_prob(x) = - sum_{i=1}^{n/2} [a_coeff * (x_{2i-1} - mu_{2i-1})^2 + b_coeff * (x_{2i} - x_{2i-1}^2)^2]
    x_j is 0-indexed internally. So, x_{2i-1} -> x_evens, x_{2i} -> x_odds
    
    From the paper:
    
    'This density is in essence the product of n/2 independent blocks, each containing a 2-d 
    Rosenbrock kernel. Unlike the Full Rosenbrock kernel, the Even Rosenbrock does maintain 
    the shape of the joint 2-d marginals as n increases. However, only a small fraction of the 
    joint distributions are curved narrow ridges, while the majority of the 2-d marginals are uncorrelated.
    '
    """
    
    def __init__(self, dim: int, a_coeff: float = DEFAULT_A_COEFF, b_coeff: float = DEFAULT_B_COEFF, 
                 mu: Union[float, torch.Tensor] = DEFAULT_MU, device: str = None):
        if dim < 2 or dim % 2 != 0:
            raise ValueError("Dimension for EvenRosenbrockTorch must be at least 2 and even.")
        super().__init__(dim, device)
        
        self.a_coeff = torch.tensor(a_coeff, device=self.device, dtype=torch.float32)
        self.b_coeff = torch.tensor(b_coeff, device=self.device, dtype=torch.float32)
        
        # mu applies to the x_{2i-1} terms. There are dim/2 such terms.
        num_mu_terms = dim // 2
        if isinstance(mu, (int, float)):
            self.mu = torch.full((num_mu_terms,), mu, device=self.device, dtype=torch.float32)
        elif isinstance(mu, torch.Tensor):
            if mu.ndim == 0:  # scalar tensor
                self.mu = torch.full((num_mu_terms,), mu.item(), device=self.device, dtype=torch.float32)
            elif mu.shape == (num_mu_terms,):
                self.mu = mu.to(device=self.device, dtype=torch.float32)
            else:
                raise ValueError(f"mu tensor must be scalar or have shape ({num_mu_terms},)")
        else:
            raise TypeError("mu must be float, int, or torch.Tensor")
        
        self._name = f"EvenRosenbrockTorch"

    def _validate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and reshape input tensor to ensure it is (batch_size, dim)."""
        if x.device != self.device:
            x = x.to(self.device)
        
        if x.ndim == 1:
            # Single point: reshape to (1, dim)
            if x.shape[0] != self.dim:
                raise ValueError(f"Expected tensor of shape ({self.dim},), got {x.shape}")
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            # Batch: ensure correct dimension
            if x.shape[1] != self.dim:
                raise ValueError(f"Expected tensor of shape (batch_size, {self.dim}), got {x.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {x.ndim}D tensor")
        
        return x

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_tensor(x)  # Ensure x is (batch_size, dim)
        
        # x_{2i-1} terms (0-indexed: x_0, x_2, ..., x_{n-2})
        x_odd_indices = x[:, 0::2] 
        # x_{2i} terms (0-indexed: x_1, x_3, ..., x_{n-1})
        x_even_indices = x[:, 1::2]
        
        term1 = self.a_coeff * (x_odd_indices - self.mu)**2  # mu is broadcasted
        term2 = self.b_coeff * (x_even_indices - x_odd_indices**2)**2
        
        log_prob = -(torch.sum(term1, dim=1) + torch.sum(term2, dim=1))
        
        # Return scalar for single point, tensor for batch
        if log_prob.shape[0] == 1:
            return log_prob.squeeze(0)
        return log_prob

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the density of the distribution at point(s) x."""
        return torch.exp(self.log_density(x))

    def get_name(self) -> str:
        return self._name

    def draw_sample(self, beta: float = 1.0) -> np.ndarray:
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        samples = self.draw_samples_torch(1, beta)
        return samples[0].cpu().numpy()
        
    def draw_samples_torch(self, n_samples: int, beta: float = 1.0) -> torch.Tensor:
        """
        Draw multiple samples using approximate conditional sampling for initialization.
        """
        samples = torch.zeros(n_samples, self.dim, device=self.device, dtype=torch.float32)
        eff_a = self.a_coeff * beta
        eff_b = self.b_coeff * beta

        # Sample x_{2i-1} terms (odd indices)
        # x_{2i-1} ~ N(mu_{2i-1}, 1/(2*eff_a))
        num_pairs = self.dim // 2
        mu_expanded = self.mu.expand(n_samples, num_pairs)  # mu applies to these terms
        
        var_odd = 1.0 / (2 * eff_a) if eff_a > 0 else 1.0
        samples_odd_indices = mu_expanded + torch.randn(n_samples, num_pairs, device=self.device, dtype=torch.float32) * torch.sqrt(var_odd)
        samples[:, 0::2] = samples_odd_indices
        
        # Sample x_{2i} terms (even indices) based on x_{2i-1}
        # x_{2i} | x_{2i-1} ~ N(x_{2i-1}^2, 1/(2*eff_b))
        x_odd_sq = samples_odd_indices**2
        var_even = 1.0 / (2 * eff_b) if eff_b > 0 else 1.0
        samples_even_indices = x_odd_sq + torch.randn(n_samples, num_pairs, device=self.device, dtype=torch.float32) * torch.sqrt(var_even)
        samples[:, 1::2] = samples_even_indices
        
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.a_coeff = self.a_coeff.to(device)
        self.b_coeff = self.b_coeff.to(device)
        self.mu = self.mu.to(device)
        return self


class HybridRosenbrockTorch(TorchTargetDistribution):
    """
    N-dimensional Hybrid Rosenbrock distribution as defined in eq. 12 of Pagani et al. (2022).
    n1 is the block length parameter; i.e. the number of variables in the block/row in the DAG
    n2 is the total number of blocks/rows in the DAG.
    Note that the final dimension of the distribution is 1 + n2 * (n1 - 1).
    
    log_prob(x) = -a(x_g1 - μ)² 
                  - b * sum_{j=1}^{n2} (x_{j,2} - x_g1^2)^2
                  - b * sum_{j=1}^{n2} sum_{i=3}^{n1} (x_{j,i} - x_{j,i-1}^2)^2
    Total dimension n = 1 + n2 * (n1 - 1).
    x_g1 is x[:,0].
    x_{j,i} are subsequent parts of x.
    """
    
    def __init__(self, n1: int, n2: int, a_coeff: float = DEFAULT_A_COEFF, b_coeff: float = DEFAULT_B_COEFF, 
                 mu: float = DEFAULT_MU, device: str = None):
        if n1 < 2:
            raise ValueError("n1 (block length parameter) must be at least 2.")
        if n2 < 1:
            raise ValueError("n2 (number of blocks) must be at least 1.")
        
        dim = 1 + n2 * (n1 - 1)
        super().__init__(dim, device)
        
        self.n1 = n1
        self.n2 = n2
        
        self.a_coeff = torch.tensor(a_coeff, device=self.device, dtype=torch.float32)
        self.b_coeff = torch.tensor(b_coeff, device=self.device, dtype=torch.float32)  # Assuming b_j,i is constant
        self.mu = torch.tensor(mu, device=self.device, dtype=torch.float32)  # mu for x_g1
        
        self._name = f"HybridRosenbrockTorch(n1={n1}, n2={n2}, a={a_coeff:.2f}, b={b_coeff:.2f}, mu={mu:.2f})"

    def _validate_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Validate and reshape input tensor to ensure it's (batch_size, dim)."""
        if x.device != self.device:
            x = x.to(self.device)
        
        if x.ndim == 1:
            # Single point: reshape to (1, dim)
            if x.shape[0] != self.dim:
                raise ValueError(f"Expected tensor of shape ({self.dim},), got {x.shape}")
            x = x.unsqueeze(0)
        elif x.ndim == 2:
            # Batch: ensure correct dimension
            if x.shape[1] != self.dim:
                raise ValueError(f"Expected tensor of shape (batch_size, {self.dim}), got {x.shape}")
        else:
            raise ValueError(f"Expected 1D or 2D tensor, got {x.ndim}D tensor")
        
        return x

    def log_density(self, x: torch.Tensor) -> torch.Tensor:
        x = self._validate_tensor(x)  # Ensure x is (batch_size, dim)
        batch_size = x.shape[0]
        
        x_g1 = x[:, 0]  # Shape: (batch_size,)
        
        # Term for x_g1
        log_prob = -self.a_coeff * (x_g1 - self.mu)**2  # Shape: (batch_size,)
        
        # Reshape the remaining variables for easier block processing
        # other_vars are x_{j,2} to x_{j,n1} for each block j
        # Dimensions of other_vars_reshaped: (batch_size, n2, n1-1)
        if self.dim > 1:  # If there are blocks beyond x_g1
            other_vars = x[:, 1:]
            other_vars_reshaped = other_vars.reshape(batch_size, self.n2, self.n1 - 1)

            # Term for x_{j,2} (first element in each block) depending on x_g1^2
            # other_vars_reshaped[:, :, 0] corresponds to x_{j,2}
            x_g1_sq = x_g1**2  # Shape: (batch_size,)
            # We need to sum over n2 blocks for this term
            term_jg1 = self.b_coeff * (other_vars_reshaped[:, :, 0] - x_g1_sq.unsqueeze(1))**2  # Unsqueeze for broadcasting
            log_prob -= torch.sum(term_jg1, dim=1)  # Sum over n2 blocks

            # Terms for x_{j,i} where i ranges from 3 to n1
            # This means k from 1 to n1-2 for 0-indexed other_vars_reshaped[:, :, k]
            # x_{j,i} (current) depends on x_{j,i-1}^2 (previous_in_block)
            if self.n1 > 2:  # If blocks have more than one variable (i.e. x_j,2, x_j,3, ...)
                for k in range(self.n1 - 2):  # k loops from 0 to n1-3
                    # x_{j,i-1} is other_vars_reshaped[:, :, k]
                    # x_{j,i} is other_vars_reshaped[:, :, k+1]
                    prev_in_block_sq = other_vars_reshaped[:, :, k]**2
                    current_in_block = other_vars_reshaped[:, :, k+1]
                    
                    term_block_internal = self.b_coeff * (current_in_block - prev_in_block_sq)**2
                    log_prob -= torch.sum(term_block_internal, dim=1)  # Sum over n2 blocks
        
        # Return scalar for single point, tensor for batch
        if log_prob.shape[0] == 1:
            return log_prob.squeeze(0)
        return log_prob

    def density(self, x: torch.Tensor) -> torch.Tensor:
        """Compute the density of the distribution at point(s) x."""
        return torch.exp(self.log_density(x))

    def get_name(self) -> str:
        return self._name

    def draw_sample(self, beta: float = 1.0) -> np.ndarray:
        """Draw a sample from the distribution (CPU implementation for compatibility)."""
        samples = self.draw_samples_torch(1, beta)
        return samples[0].cpu().numpy()

    def draw_samples_torch(self, n_samples: int, beta: float = 1.0) -> torch.Tensor:
        """
        This implements Algorithm 1 from the paper for direct sampling.
        """
        samples = torch.zeros(n_samples, self.dim, device=self.device, dtype=torch.float32)
        
        eff_a = self.a_coeff * beta
        eff_b = self.b_coeff * beta  # Assuming constant b_j,i

        # 1. Sample x_g1 (X1 in Algorithm 1)
        # x_g1 ~ N(mu, 1 / (2*eff_a))
        var_g1 = 1.0 / (2 * eff_a) if eff_a > 0 else 1.0
        std_g1 = torch.sqrt(torch.scalar_tensor(var_g1, device=self.device, dtype=torch.float32))
        samples[:, 0] = self.mu + torch.randn(n_samples, device=self.device, dtype=torch.float32) * std_g1
        
        if self.dim == 1:  # Only x_g1
            return samples

        # 2. Sample block variables
        # X_{j,i} | X_{j,i-1} ~ N(X_{j,i-1}^2, 1 / (2*eff_b))
        # (or X_{j,2} | x_g1 ~ N(x_g1^2, 1 / (2*eff_b)) )
        var_block = 1.0 / (2 * eff_b) if eff_b > 0 else 1.0
        std_block = torch.sqrt(torch.scalar_tensor(var_block, device=self.device, dtype=torch.float32))
        
        current_flat_idx = 1
        for j in range(self.n2):  # Loop over blocks
            # First variable in block (x_{j,2}) depends on x_g1^2
            prev_var_sq = samples[:, 0]**2  # x_g1^2
            samples[:, current_flat_idx] = prev_var_sq + torch.randn(n_samples, device=self.device, dtype=torch.float32) * std_block
            current_flat_idx += 1
            
            # Subsequent variables in the block (x_{j,3} to x_{j,n1})
            for _ in range(self.n1 - 2):  # Loop for remaining n1-2 variables in the block
                # The previous variable was samples[:, current_flat_idx - 1]
                prev_var_sq = samples[:, current_flat_idx - 1]**2
                samples[:, current_flat_idx] = prev_var_sq + torch.randn(n_samples, device=self.device, dtype=torch.float32) * std_block
                current_flat_idx += 1
                
        return samples

    def to(self, device):
        """Move the distribution to a specific device."""
        super().to(device)
        self.a_coeff = self.a_coeff.to(device)
        self.b_coeff = self.b_coeff.to(device)
        self.mu = self.mu.to(device)
        return self 