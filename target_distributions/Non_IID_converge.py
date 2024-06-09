import numpy as np
from interfaces import TargetDistribution
from scipy.stats import gamma, beta
from scipy.stats import norm, multivariate_normal
from numba import njit

class NONIIDGamma(TargetDistribution):
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
        if len(x) != self.dim:
            raise ValueError(
                "Dimension of x must be equal to the specified dimensions.")
        product = 1
        for i in range(1, self.dim + 1):
            ith_shape = self.shape + 1 / (5*i)
            ith_scale = self.scale - 1 / (5*i)
            pdf_i = gamma.pdf(x[i - 1], a=ith_shape, b=ith_scale)
            product *= pdf_i
        return product


class NONIIDBeta(TargetDistribution):
    """Class for a product of iid Beta distributions."""

    def __init__(self, dimension, alpha=2, beta=5):
        """Initialize the product of iid Beta distribution.

        Parameters:
        alpha (float): The alpha (shape1) parameter of the Beta distribution.
        beta (float): The beta (shape2) parameter of the Beta distribution.
        dimension (int): The number of dimensions.
        """
        super().__init__(dimension)
        self.name = "NONIIDBeta"
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
            raise ValueError(
                "Dimension of x must be equal to the specified dimensions.")
        product = 1
        for i in range(1, self.dim + 1):
            ith_alpha = self.alpha + 1 / (5*i)
            ith_beta = self.beta - 1 / (5*i)
            pdf_i = beta.pdf(x[i - 1], a=ith_alpha, b=ith_beta)
            product *= pdf_i
        return product


class NONIIDGaussian(TargetDistribution):
    """Class for a product of iid Beta distributions."""

    def __init__(self, dim, mean, cov):
        """Initialize the product of iid Beta distribution.

        Parameters:
        alpha (float): The alpha (shape1) parameter of the Beta distribution.
        beta (float): The beta (shape2) parameter of the Beta distribution.
        dimension (int): The number of dimensions.
        """
        super().__init__(dim)
        self.name = "NONIIDGaussian"
        self.mean = mean
        self.cov = cov

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
            raise ValueError(
                "Dimension of x must be equal to the specified dimensions.")
        product = 1

        for i in range(1, self.dim + 1):
            ith_mean = 1 /(5*i)
            ith_var = 1
            pdf_i = norm.pdf(x[i - 1], loc=ith_mean, scale=ith_var)
            product *= pdf_i
        return product
