import numpy as np
from interfaces import TargetDistribution
from scipy.stats import gamma, beta

class IIDGamma(TargetDistribution):
    """Class for a product of iid gamma distributions."""

    def __init__(self, dimension, shape=2, scale=3):
        """Initialize the product of iid Gamma distribution.
        
        Parameters:
        shape (float): The shape parameter of the Gamma distribution.
        scale (float): The scale parameter of the Gamma distribution.
        dimensions (int): The number of dimensions.
        """
        super().__init__(dimension)
        self.name = "IIDGamma"
        self.shape = shape
        self.scale = scale

    def get_name(self):
        """
        Return the name of the target distribution as a string.
        """
        return self.name
    
    def density(self, x):
        """
        Evaluate the density at a point x in d-dimensional space.
        
        Parameters:
        x (array-like): A point in d-dimensional space.
        
        Returns:
        float: The density evaluated at x.
        """
        return np.prod(gamma.pdf(x, a=self.shape, scale=self.scale))
    
    def draw_sample(self, beta=1.0):
        """Draw a sample from the target distribution. This is meant to be a cheap heuristic
        used for constructing the temperature ladder in parallel tempering.
        Do not use this to draw samples in an actual Metropolis algorithm."""
        adjusted_shape = self.shape * beta
        return gamma.rvs(a=adjusted_shape, scale=self.scale, size=self.dim)
    

class IIDBeta(TargetDistribution):
    """Class for a product of iid Beta distributions."""

    def __init__(self, dimension, alpha=2, beta=3):
        """Initialize the product of iid Beta distribution.
        
        Parameters:
        alpha (float): The alpha (shape1) parameter of the Beta distribution.
        beta (float): The beta (shape2) parameter of the Beta distribution.
        dimension (int): The number of dimensions.
        """
        super().__init__(dimension)
        self.name = "IIDBeta"
        self.alpha = alpha
        self.beta = beta

    def get_name(self):
        """
        Return the name of the target distribution as a string.
        """
        return self.name
    
    def density(self, x):
        """
        Evaluate the density at a point x in d-dimensional space.
        
        Parameters:
        x (array-like): A point in d-dimensional space.
        
        Returns:
        float: The density evaluated at x.
        """
        if len(x) != self.dim:
            raise ValueError("Dimension of x must be equal to the specified dimensions.")
        
        return np.prod(beta.pdf(x, a=self.alpha, b=self.beta))
    
    def draw_sample(self, beta_temp=1.0):
        """Draw a sample from the target distribution. This is meant to be a cheap heuristic
        used for constructing the temperature ladder in parallel tempering.
        Do not use this to draw samples in an actual Metropolis algorithm.
        
        Parameters:
        beta_temp (float): The inverse temperature parameter.
        
        Returns:
        numpy.ndarray: A sample from the target density.
        """
        if beta_temp <= 0:
            raise ValueError("Inverse temperature parameter beta_temp must be positive.")

        # Adjust the alpha and beta parameters by the inverse temperature beta_temp
        adjusted_alpha = self.alpha * beta_temp
        adjusted_beta = self.beta * beta_temp

        # Draw samples from the adjusted Beta distribution
        return beta.rvs(a=adjusted_alpha, b=adjusted_beta, size=self.dim)
