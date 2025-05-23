# Target Distributions Directory

This directory `target_distributions/` implements various target probability distributions for Monte Carlo sampling algorithms. Each distribution class inherits from the `TargetDistribution` interface defined in `montecarlo/interfaces/target.py` and provides implementations for density evaluation and sample generation.

## Overview

The target distributions serve as the probability distributions that the Monte Carlo algorithms (Random Walk Metropolis and Parallel Tempering) attempt to sample from. They range from simple unimodal distributions to complex multimodal distributions, each presenting different challenges for sampling algorithms.

## Files Documentation

### `__init__.py`
**Purpose**: Module initialization file that exports all distribution classes.
**Exports**: 
- All classes from `multimodal`, `multivariate_normal`, `hypercube`, and `iid_product` modules
- Provides a clean interface for importing distributions from this package

### `multivariate_normal.py`
**Purpose**: Implements multivariate normal (Gaussian) distribution.
**Classes**:
- `MultivariateNormal`: A d-dimensional Gaussian distribution
**Key Features**:
- Default configuration: zero mean vector and identity covariance matrix
- Supports custom mean vector and covariance matrix specification
- Handles both 1D and multi-dimensional density evaluation
- Includes temperature-scaled sampling for parallel tempering (via `beta` parameter)
**Methods**:
- `density(x)`: Evaluates PDF at point x
- `density_1d(x)`: Specialized 1D density evaluation
- `draw_sample(beta=1)`: Generates samples with optional temperature scaling

### `hypercube.py`
**Purpose**: Implements uniform distribution over a hypercube domain.
**Classes**:
- `Hypercube`: Uniform distribution over [left_boundary, right_boundary]^d
**Key Features**:
- Configurable boundary values (default: [0,1]^d)
- Product structure for multi-dimensional evaluation
- Simple uniform sampling within boundaries
**Methods**:
- `density(x)`: Returns 1 if x is within boundaries, 0 otherwise
- `density_1d(x)`: Individual component density evaluation
- `draw_sample()`: Uniform random sampling within hypercube

### `multimodal.py`
**Purpose**: Implements complex multimodal distributions for testing algorithm performance.
**Classes**:
1. `ThreeMixtureDistribution`: Mixture of three multivariate Gaussians
   - Three modes positioned at (-15,0,...,0), (0,0,...,0), and (15,0,...,0)
   - Equal mixing weights (1/3 each)
   - Optional scaling factors for different dimensions
   - Useful for testing mode-switching capabilities

2. `RoughCarpetDistribution`: Product of 1D three-mode distributions
   - Each dimension has three modes at -15, 0, 15
   - Unequal mixing weights: [0.5, 0.3, 0.2]
   - Creates a "rough carpet" structure in high dimensions
   - Optional coordinate-wise scaling

**Key Features**:
- Both support optional scaling for varying difficulty across dimensions
- Temperature-scaled sampling for parallel tempering algorithms
- Designed to challenge sampling algorithms with multiple modes

### `iid_product.py`
**Purpose**: Implements distributions as products of independent identical distributions.
**Classes**:
1. `IIDGamma`: Product of independent Gamma distributions
   - Configurable shape and scale parameters
   - Temperature scaling via shape parameter adjustment

2. `IIDBeta`: Product of independent Beta distributions
   - Configurable alpha and beta parameters
   - Temperature scaling via parameter multiplication

**Key Features**:
- Dimension scaling through independent components
- Natural support for parallel tempering temperature schedules
- Non-Gaussian alternatives for testing robustness

### `IID_Gamma_Beta.py`
**Purpose**: Duplicate implementation of IID Gamma and Beta distributions.
**Note**: This appears to be a duplicate of `iid_product.py` with identical implementations of `IIDGamma` and `IIDBeta` classes. This file may be redundant and could potentially be removed to avoid confusion.

## Common Interface

All distribution classes implement the `TargetDistribution` interface with these key methods:
- `density(x)`: Evaluate probability density at point x
- `draw_sample(beta=1)`: Generate samples (with optional temperature parameter)
- `get_name()`: Return string identifier for the distribution

The `beta` parameter in `draw_sample()` represents inverse temperature for parallel tempering, where higher values of beta (colder temperatures) create more concentrated distributions around modes.