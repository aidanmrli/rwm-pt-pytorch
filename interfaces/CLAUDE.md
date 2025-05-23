# Interfaces Directory

This directory `interfaces/` contains abstract base classes and interfaces that define the core contracts for the Monte Carlo simulation framework. These interfaces ensure consistent APIs across different implementations of target distributions, MCMC algorithms, and simulation controllers.

## Overview

The interfaces establish a modular architecture where:
- Target distributions can be easily swapped and extended
- Different MCMC algorithms can be implemented with consistent interfaces
- Simulation execution and analysis follow standardized patterns
- Code reusability and extensibility are maximized

## Files Documentation

### `__init__.py`
**Purpose**: Module initialization file that exports the core interfaces.
**Exports**:
- `MHAlgorithm`: Abstract base class for Metropolis-Hastings algorithms
- `MCMCSimulation`: Simulation controller and analysis framework
- `TargetDistribution`: Abstract base class for target probability distributions

### `target.py`
**Purpose**: Defines the abstract interface for target probability distributions.
**Classes**:
- `TargetDistribution`: Abstract base class that all target distributions must inherit from

**Interface Requirements**:
- `__init__(dimension)`: Initialize with specified dimensionality
- `density(x)`: **Must implement** - Evaluate probability density at point x
- `draw_sample(beta=1.0)`: **Must implement** - Generate samples (used for temperature ladder construction in parallel tempering)

**Key Design Principles**:
- The `draw_sample()` method is explicitly noted as a "cheap heuristic" for parallel tempering setup
- Not intended for actual MCMC sampling (that's handled by the algorithms)
- The `beta` parameter enables temperature scaling for parallel tempering

### `metropolis.py`
**Purpose**: Defines the abstract interface for Metropolis-Hastings algorithms.
**Classes**:
- `MHAlgorithm`: Abstract base class for all MH algorithm implementations

**Key Attributes**:
- `dim`: Dimensionality of the problem
- `var`: Proposal variance parameter
- `target_dist`: Reference to the target distribution
- `chain`: Stores the complete Markov chain of samples
- `symmetric`: Boolean flag for symmetric proposals
- `num_acceptances`: Counter for accepted proposals
- `acceptance_rate`: Current acceptance rate
- `target_density`: Cached reference to target density function

**Interface Requirements**:
- `step()`: **Must implement** - Execute one MCMC step
- `get_name()`: **Must implement** - Return algorithm identifier string

**Provided Methods**:
- `reset()`: Reset chain to initial state
- `get_curr_state()`: Return current chain position
- `set_curr_state(state)`: Update current chain position

**Key Design Features**:
- Stores complete chain history for analysis
- Tracks acceptance statistics automatically
- Supports both symmetric and asymmetric proposals
- Designed for subclassing with specific acceptance probability calculations

### `simulation.py`
**Purpose**: Provides a comprehensive simulation controller and analysis framework.
**Classes**:
- `MCMCSimulation`: Orchestrates MCMC execution and provides analysis tools

**Constructor Parameters**:
- `dim`: Problem dimensionality
- `sigma`: Proposal variance parameter
- `num_iterations`: Number of MCMC steps to run
- `algorithm`: MHAlgorithm class (not instance)
- `target_dist`: TargetDistribution instance
- `symmetric`: Proposal symmetry flag
- `seed`: Random seed for reproducibility
- `beta_ladder`: Temperature schedule for parallel tempering
- `swap_acceptance_rate`: PT swap acceptance tracking

**Core Functionality**:

#### Simulation Control:
- `generate_samples()`: Execute the MCMC simulation with progress tracking
- `reset()`: Reset simulation state for re-running
- `has_run()`: Check if simulation has been executed

#### Analysis Methods:
- `acceptance_rate()`: Calculate overall acceptance rate
- `expected_squared_jump_distance()`: Compute ESJD for standard MCMC
- `pt_expected_squared_jump_distance()`: Compute ESJD for parallel tempering

#### Visualization:
- `traceplot(single_dim=False, show=False)`: Generate traceplots of the Markov chain
  - Supports single or multi-dimensional plotting
  - Automatically saves high-resolution figures
  - Configurable display options

- `samples_histogram(num_bins=50, axis=0, show=False)`: Create sample histograms with target overlay
  - Overlays true target density (red dashed line)
  - Focuses on specified coordinate axis
  - Validates convergence and correctness

**Key Design Features**:
- Automatic progress tracking with tqdm
- Integrated error checking and validation
- Standardized file naming for saved plots
- Support for both standard MCMC and parallel tempering analysis
- High-resolution plot generation for publication
- Modular algorithm instantiation within simulation framework

## Architecture Benefits

This interface-based design enables:
1. **Easy Algorithm Comparison**: Swap different MCMC algorithms with identical analysis
2. **Target Distribution Flexibility**: Test algorithms on various probability distributions  
3. **Consistent Analysis Pipeline**: Standardized metrics and visualization across all experiments
4. **Extensibility**: Add new algorithms or distributions without modifying existing code
5. **Reproducibility**: Built-in seeding and standardized experiment structure