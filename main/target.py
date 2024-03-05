class TargetDistribution:
    """General class for target distributions."""

    def __init__(self, dimension):
        self.dim = dimension

    def density(self, x):
        """Compute the density of the distribution at a given point x."""
        raise NotImplementedError("Subclasses must implement the density method.")

    def grad_log_density(self, x):
        """Compute the gradient of the log density of the distribution at a given point x."""
        raise NotImplementedError("Subclasses must implement the grad_log_density method.")