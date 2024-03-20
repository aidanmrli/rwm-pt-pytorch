import numpy as np
from main import TargetDistribution
from scipy.stats import norm, multivariate_normal

class MultivariateNormal(TargetDistribution):
    """
    Class representing a multivariate normal distribution.
    Default has zero mean and identity covariance matrix.
    """

    def __init__(self, dim, mean=None, cov=None):
        """
        Initializes the MultivariateNormal distribution with a mean vector and covariance matrix.

        Args:
            mean (np.ndarray): Mean vector of the distribution.
            cov (np.ndarray): Covariance matrix of the distribution.
        """
        super().__init__(dim)  # Dimension based on mean vector size
        if mean is None:
            mean = np.zeros(dim)
        if cov is None:
            cov = np.eye(dim)
        self.mean = mean
        self.cov = cov

    def density_1d(self, x):
        """
        Evaluates the probability density function (PDF) at a point x. Assumes that the distribution is 1D.
        If evaluating a single component of a multi-dimensional x, the use of this function assumes that 
        the other components are independent.
        Args:
            x (np.ndarray): Point in the multivariate normal distribution domain with the same dimension as the mean vector.

        Returns:
            float: The value of the PDF at the point x.
        """
        return norm.pdf(x, loc=self.mean, scale=np.sqrt(self.cov))
    
    def density(self, x):
        """
        Evaluates the probability density function (PDF) at a point x.

        Args:
            x (np.ndarray): Point in the multivariate normal distribution domain with the same dimension as the mean vector.

        Returns:
            float: The value of the PDF at the point x.
        """
        if x is float or len(x) == 1:
            return self.density_1d(x)
        # x_centered = x - self.mean
        # inv_cov = np.linalg.inv(self.cov)
        # quad_form = np.dot(x_centered.T, np.dot(inv_cov, x_centered))
        # return (1 / np.sqrt((2 * np.pi) ** self.dim * np.linalg.det(self.cov))) * np.exp(-0.5 * quad_form)
        return multivariate_normal.pdf(x, mean=self.mean, cov=self.cov)

    def grad_log_density(self, x):
        """
        Calculates the gradient of the log-density function at a point x.

        Args:
            x (np.ndarray): Point in the multivariate normal distribution domain with the same dimension as the mean vector.

        Returns:
            np.ndarray: The gradient of the log-density function at the point x.
        """
        x_centered = x - self.mean
        inv_cov = np.linalg.inv(self.cov)
        return -inv_cov.dot(x_centered)
