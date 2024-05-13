import numpy as np
from interfaces import TargetDistribution

class Hypercube(TargetDistribution):
    """
    Class representing the hypercube target density.
    """

    def __init__(self, dim, left_boundary=0.0, right_boundary=1.0):
        """
        Initializes the Hypercube target density with the dimension.

        Args:
            dim (int): Dimension of the hypercube.
        """
        super().__init__(dim)  # Dimension based on input
        self.left_boundary = left_boundary
        self.right_boundary = right_boundary

    def density_1d(self, x):
        """
        Evaluates the individual component (1d) density function at a point x.

        Args:
            x (np.ndarray): Point in the hypercube domain.

        Returns:
            float: The value of the PDF at the point x.
        """
        if x < self.left_boundary or x > self.right_boundary:
            return 0
        return 1
    
    def density(self, x):
        """
        Evaluates the probability density function (PDF) at a point x.

        Args:
            x (np.ndarray):

        Returns:
            float: The value of the PDF at the point x.
        """
        return np.prod([self.density_1d(x[i]) for i in range(self.dim)])

    def draw_sample(self):
        """
        Draws a sample from the hypercube target density.

        Returns:
            np.ndarray: A sample from the hypercube target density.
        """
        return np.random.uniform(self.left_boundary, self.right_boundary, self.dim)