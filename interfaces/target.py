class TargetDistribution:
    """General interface for target distributions."""

    def __init__(self, dimension):
        self.dim = dimension

    def density(self, x):
        """Compute the density of the distribution at a given point x."""
        raise NotImplementedError("Subclasses must implement the density method.")

    def draw_sample(self, beta=1.0):
        """Draw a sample from the target distribution. This is meant to be a cheap heuristic
        used for constructing the temperature ladder in parallel tempering.
        Do not use this to draw samples in an actual Metropolis algorithm."""
        raise NotImplementedError("Subclasses must implement the draw_sample method.")