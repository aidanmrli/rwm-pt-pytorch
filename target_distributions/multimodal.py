import numpy as np
from main import TargetDistribution
from scipy.stats import multivariate_normal

class MultimodalDensityNew(TargetDistribution):
    """Class for a multimodal target distribution with three modes."""

    def __init__(self, dimension, scaling=False):
        super().__init__(dimension)
        if scaling:
            self.scaling_factors = np.random.uniform(0.000001, 2, self.dim)    # Randomly sample scaling factors, must have mean 1
    
    # def density_1d(self, x):
    #     """Compute the density of a multimodal 1D distribution with three modes."""
    #     mode1 = np.exp(-0.5 * (x - (-10))**2) / np.sqrt(2 * np.pi)
    #     mode2 = np.exp(-0.5 * (x - 0)**2) / np.sqrt(2 * np.pi)
    #     mode3 = np.exp(-0.5 * (x - 10)**2) / np.sqrt(2 * np.pi)
    #     return 0.5 * mode1 + 0.3 * mode2 + 0.2 * mode3
    
    def density(self, x):
        """Compute the density of the multimodal distribution at a given point x.
        This density function has three modes: one at (-5, 0, 0, ..., 0), one at (0, 0, 0, ..., 0), and one at (5, 0, 0, ..., 0).
        Args:
            x (np.ndarray): A datapoint.

        Returns:
            float: The density value for the input data point.
        """
        means = [np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)]
        means[0][0], means[2][0] = -5 / np.sqrt(self.dim), 5 / np.sqrt(self.dim)

        covs = [np.eye(self.dim) / np.sqrt(self.dim), np.eye(self.dim) / np.sqrt(self.dim), np.eye(self.dim) / np.sqrt(self.dim)]

        # Calculate the density for each mode and sum them
        density = 0
        for mean, cov in zip(means, covs):
            density += 1/3 * multivariate_normal.pdf(x, mean=mean, cov=cov)

        # print("this is x")
        # print(x)
        # print("this is the density")
        # print(density)
        return density
    

    def grad_log_density(self, x):
        """Compute the gradient of the log density of the multimodal distribution at a given point x."""
        grad_log_probs = np.array([component.grad_log_density(x) for component in self.components])
        weights = np.exp([component.log_density(x) for component in self.components])
        weights /= np.sum(weights)
        return np.sum(grad_log_probs * weights[:, np.newaxis], axis=0)



class MultimodalDensityIID(TargetDistribution):
    """Class for 'rough carpet' multimodal target distributions that are products of 1D multimodal distributions."""

    def __init__(self, dimension, scaling=False):
        super().__init__(dimension)
        if scaling:
            self.scaling_factors = np.random.uniform(0.000001, 2, self.dim)    # Randomly sample scaling factors, must have mean 1
    
    def density_1d(self, x):
        """Compute the density of a multimodal 1D distribution with three modes."""
        mode1 = np.exp(-0.5 * (x - (-5))**2) / np.sqrt(2 * np.pi)
        mode2 = np.exp(-0.5 * (x - 0)**2) / np.sqrt(2 * np.pi)
        mode3 = np.exp(-0.5 * (x - 5)**2) / np.sqrt(2 * np.pi)
        return 0.5 * mode1 + 0.3 * mode2 + 0.2 * mode3
    
    def density(self, x):
        """Compute the density of the multimodal distribution at a given point x."""
        if isinstance(x, (int, float)) or len(x) == 1:
            if hasattr(self, 'scaling_factors'):
                return self.scaling_factors[0] * self.density_1d(self.scaling_factors[0] * x)
            else:
                return self.density_1d(x)
            
        elif hasattr(self, 'scaling_factors'):
                return np.prod([self.scaling_factors[i] * self.density_1d(self.scaling_factors[i] * x[i])
                            for i in range(self.dim)])
        else:  
            return np.prod([self.density_1d(x[i]) for i in range(self.dim)])

    def grad_log_density(self, x):
        """Compute the gradient of the log density of the multimodal distribution at a given point x."""
        grad_log_probs = np.array([component.grad_log_density(x) for component in self.components])
        weights = np.exp([component.log_density(x) for component in self.components])
        weights /= np.sum(weights)
        return np.sum(grad_log_probs * weights[:, np.newaxis], axis=0)

