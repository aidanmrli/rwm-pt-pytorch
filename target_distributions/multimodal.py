import numpy as np
from interfaces import TargetDistribution
from scipy.stats import multivariate_normal

class ThreeMixtureDistribution(TargetDistribution):
    """Class for a multimodal target distribution with three modes:
        one at (-c, 0, 0, ..., 0), 
        one at (0, 0, 0, ..., 0), 
        and one at (c, 0, 0, ..., 0) where c is some constant
        that is defined by the target distribution init method."""

    def __init__(self, dimension, scaling=False):
        """Initialize the multimodal distribution with three modes.
        Set the locations of the means and the covariance matrices for each mode."""
        super().__init__(dimension)
        self.name = "ThreeMixture"
        self.means = [np.zeros(self.dim), np.zeros(self.dim), np.zeros(self.dim)]
        self.means[0][0], self.means[2][0] = -15, 15

        ### Choose which covariance matrix to use!
        # self.covs = [np.eye(self.dim) / np.sqrt(self.dim), np.eye(self.dim) / np.sqrt(self.dim), np.eye(self.dim) / np.sqrt(self.dim)]
        # self.covs = [np.eye(self.dim) * 0.6, np.eye(self.dim) * 1.0, np.eye(self.dim) * 1.4]
        self.covs = [np.eye(self.dim), np.eye(self.dim), np.eye(self.dim)]
        if scaling:
            self.scaling_factors = np.random.uniform(0.000001, 2, self.dim)    # Randomly sample scaling factors, must have mean 1
            for i in range(len(self.covs)):
                self.covs[i] *= self.scaling_factors

    def get_name(self):
        """
        Return the name of the target distribution as a string.
        """
        return self.name
    
    def density(self, x):
        """Compute the density of the multimodal distribution at a given point x.
        This density function has three modes: one at (-c, 0, 0, ..., 0), 
        one at (0, 0, 0, ..., 0), and one at (c, 0, 0, ..., 0) where c is some constant
        that is defined by the target distribution init method.
        Args:
            x (np.ndarray): A datapoint.

        Returns:
            float: The density value for the input data point.
        """
        # Calculate the density for each mode and sum them
        density = 0
        for mean, cov in zip(self.means, self.covs):
            density += 1/3 * multivariate_normal.pdf(x, mean=mean, cov=cov)

        return density
    
    def draw_sample(self, beta=1):
        """Draw a sample from the target distribution. This is meant to be a cheap heuristic
        used for constructing the temperature ladder in parallel tempering.
        Do not use this to draw samples in an actual Metropolis algorithm."""
        # pick a mode at random and sample from the mode
        random_integer = np.random.randint(0, 3)  
        target_mean, target_cov = self.means[random_integer], self.covs[random_integer]
        
        return np.random.multivariate_normal(target_mean, target_cov / beta)
    

class RoughCarpetDistribution(TargetDistribution):
    """Class for 'rough carpet' multimodal target distributions that are products of 1D multimodal distributions."""

    def __init__(self, dimension, scaling=False):
        super().__init__(dimension)
        self.name = "RoughCarpet"
        self.modes = [-5, 0, 5]
        self.weights = [0.5, 0.3, 0.2]
        if scaling:  # Randomly sample scaling factors, distribution must have expectation 1
            self.scaling_factors = np.random.uniform(0.000001, 2, self.dim)
    
    def get_name(self):
        """
        Return the name of the target distribution as a string.
        """
        return self.name
    
    def density_1d(self, x):
        """Compute the density of a multimodal 1D distribution with three modes."""
        mode0 = np.exp(-0.5 * (x - self.modes[0])**2) / np.sqrt(2 * np.pi)
        mode1 = np.exp(-0.5 * (x - self.modes[1])**2) / np.sqrt(2 * np.pi)
        mode2 = np.exp(-0.5 * (x - self.modes[2])**2) / np.sqrt(2 * np.pi)
        return self.weights[0] * mode0 + self.weights[1] * mode1 + self.weights[2] * mode2
    
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

    def draw_sample(self, beta=1):
        """Draw a sample from the target distribution. This is meant to be a cheap heuristic
        used for constructing the temperature ladder in parallel tempering.
        Do not use this to draw samples in an actual Metropolis algorithm."""
        sample = np.zeros(self.dim)

        for i in range(self.dim):
            mean_value = np.random.choice(self.modes, p=self.weights)
            sample[i] = np.random.normal(mean_value, 1 / beta)
        
        return sample