# Proposal Distributions (`proposal_distributions`)

This directory contains the implementation of various proposal distributions for use in MCMC algorithms, particularly `RandomWalkMH_GPU_Optimized`.

## Structure

-   `base.py`: Defines the abstract base class `ProposalDistribution`. All specific proposal distributions inherit from this class.
-   `normal.py`: Implements `NormalProposal` (multivariate normal distribution).
-   `laplace.py`: Implements `LaplaceProposal` (multivariate Laplace distribution).
-   `uniform.py`: Implements `UniformRadiusProposal` (uniform distribution within an n-ball).
-   `__init__.py`: Exports the main classes for easy importing from this package (e.g., `from proposal_distributions import NormalProposal`).

## `ProposalDistribution` Base Class

The `ProposalDistribution` class in `base.py` provides a common interface:

-   `__init__(self, dim, beta, device, dtype, rng_generator)`: Constructor.
-   `sample(self, n_samples)`: Abstract method to generate `n_samples` proposal increments.
-   `get_name(self)`: Abstract method to return the name of the proposal.
-   `sample_into(self, n_samples, output_tensor)`: Optional method to sample directly into a pre-allocated tensor.

## Usage

These proposal distributions are primarily used by `MCMCSimulation_GPU` (in `interfaces/simulation_gpu.py`) which instantiates them based on a configuration dictionary, and by `RandomWalkMH_GPU_Optimized` (in `algorithms/rwm_gpu_optimized.py`) which uses the proposal object to generate candidate steps.

The `proposal_config` dictionary typically looks like:
```python
proposal_config = {
    'name': 'Normal',  # or 'Laplace', 'UniformRadius'
    'params': {
        # Parameters specific to the proposal type
        'base_variance_scalar': 0.1, # For NormalProposal
        # 'base_variance_vector': [0.1, 0.2], # For LaplaceProposal
        # 'base_radius': 1.0, # For UniformRadiusProposal
    }
}
``` 