# Interfaces Directory

This directory `interfaces/` contains abstract base classes and interfaces that define the core contracts for the Monte Carlo simulation framework. These interfaces ensure consistent APIs across different implementations of target distributions, MCMC algorithms, and simulation controllers, with support for both CPU and GPU acceleration.

## Overview

The interfaces establish a modular architecture where:
- Target distributions can be easily swapped and extended (both CPU and GPU implementations)
- Different MCMC algorithms can be implemented with consistent interfaces
- Simulation execution and analysis follow standardized patterns with GPU acceleration options
- Code reusability and extensibility are maximized across CPU and GPU environments
- Performance benchmarking and optimization are integrated into the framework

## Files Documentation

### `__init__.py`
**Purpose**: Module initialization file that exports the core interfaces.
**Exports**:
- `MHAlgorithm`: Abstract base class for Metropolis-Hastings algorithms
- `MCMCSimulation`: CPU-based simulation controller and analysis framework
- `MCMCSimulation_GPU`: GPU-accelerated simulation controller with performance optimization
- `TargetDistribution`: Abstract base class for target probability distributions (CPU)
- `TorchTargetDistribution`: PyTorch-native abstract base class for GPU-accelerated target distributions

### `target.py`
**Purpose**: Defines the basic abstract interface for target probability distributions (CPU implementation).
**Classes**:
- `TargetDistribution`: Abstract base class that all CPU-based target distributions must inherit from

**Interface Requirements**:
- `__init__(dimension)`: Initialize with specified dimensionality
- `density(x)`: **Must implement** - Evaluate probability density at point x
- `draw_sample(beta=1.0)`: **Must implement** - Generate samples (used for temperature ladder construction in parallel tempering)

**Key Design Principles**:
- The `draw_sample()` method is explicitly noted as a "cheap heuristic" for parallel tempering setup
- Not intended for actual MCMC sampling (that's handled by the algorithms)
- The `beta` parameter enables temperature scaling for parallel tempering

### `target_torch.py`
**Purpose**: Defines PyTorch-native interface for GPU-accelerated target probability distributions.
**Classes**:
- `TorchTargetDistribution`: Abstract base class for GPU-accelerated target distributions

**Key Attributes**:
- `dim`: Dimensionality of the distribution
- `device`: PyTorch device (auto-detected or manually specified)

**Interface Requirements**:
- `__init__(dimension, device=None)`: Initialize with dimensionality and optional device specification
- `density(x)`: **Must implement** - Compute density for single point or batch (supports GPU tensors)
- `log_density(x)`: **Must implement** - Compute log density for numerical stability (supports GPU tensors)
- `get_name()`: **Must implement** - Return distribution name as string
- `draw_sample(beta=1.0)`: **Should implement** - CPU compatibility method for temperature ladder construction

**Key Design Features**:
- Automatic GPU device detection and management
- Support for both single-point and batch computations
- Separate `density()` and `log_density()` methods for numerical stability
- `to(device)` method for device transfers
- Backward compatibility with CPU-based interfaces

### `metropolis.py`
**Purpose**: Defines the abstract interface for Metropolis-Hastings algorithms (supports both CPU and GPU target distributions).
**Classes**:
- `MHAlgorithm`: Abstract base class for all MH algorithm implementations

**Key Attributes**:
- `dim`: Dimensionality of the problem
- `var`: Proposal variance parameter
- `target_dist`: Reference to target distribution (TargetDistribution or TorchTargetDistribution)
- `chain`: Stores the complete Markov chain of samples (list of numpy arrays)
- `symmetric`: Boolean flag for symmetric proposals
- `num_acceptances`: Counter for accepted proposals
- `acceptance_rate`: Current acceptance rate
- `target_density`: Cached reference to target density function

**Interface Requirements**:
- `step()`: **Must implement** - Execute one MCMC step
- `get_name()`: **Must implement** - Return algorithm identifier string

**Provided Methods**:
- `reset()`: Reset chain to initial state
- `get_curr_state()`: Return current chain position (last element)
- `set_curr_state(state)`: Update current chain position

**Key Design Features**:
- Stores complete chain history for analysis
- Tracks acceptance statistics automatically
- Supports both symmetric and asymmetric proposals
- Compatible with both CPU and GPU target distributions
- Designed for subclassing with specific acceptance probability calculations

### `simulation.py`
**Purpose**: Provides a comprehensive CPU-based simulation controller and analysis framework.
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
- `burn_in`: Number of initial samples to discard (default: 0)

**Core Functionality**:

#### Simulation Control:
- `generate_samples()`: Execute the MCMC simulation with progress tracking via tqdm
- `reset()`: Reset simulation state for re-running
- `has_run()`: Check if simulation has been executed

#### Analysis Methods:
- `acceptance_rate()`: Calculate overall acceptance rate (excluding burn-in)
- `expected_squared_jump_distance()`: Compute ESJD for standard MCMC (excluding burn-in)
- `pt_expected_squared_jump_distance()`: Compute ESJD for parallel tempering

#### Visualization:
- `traceplot(single_dim=False, show=False)`: Generate traceplots of the Markov chain
  - Supports single or multi-dimensional plotting
  - Automatically saves high-resolution figures to `images/publishing/`
  - Configurable display options

- `samples_histogram(num_bins=50, axis=0, show=False)`: Create sample histograms with target overlay
  - Overlays true target density (red dashed line)
  - Focuses on specified coordinate axis
  - Validates convergence and correctness
  - Automatically saves to `images/publishing/`

**Key Design Features**:
- Automatic progress tracking with tqdm
- Integrated burn-in handling for all metrics
- Standardized file naming for saved plots
- Support for both standard MCMC and parallel tempering analysis
- High-resolution plot generation (300 DPI) for publication
- Modular algorithm instantiation within simulation framework

### `simulation_gpu.py`
**Purpose**: Provides GPU-accelerated simulation controller with performance optimization and benchmarking capabilities.
**Classes**:
- `MCMCSimulation_GPU`: GPU-accelerated MCMC simulation framework

**Constructor Parameters**:
- `dim`: Dimensionality of the problem
- `sigma`: Proposal variance parameter
- `num_iterations`: Number of MCMC steps to run
- `algorithm`: MHAlgorithm class (not instance)
- `target_dist`: TargetDistribution or TorchTargetDistribution instance
- `symmetric`: Proposal symmetry flag
- `seed`: Random seed for reproducibility
- `beta_ladder`: Temperature schedule for parallel tempering
- `swap_acceptance_rate`: PT swap acceptance tracking
- `device`: GPU device specification ('cuda', 'cpu', or None for auto-detection)
- `pre_allocate`: Whether to pre-allocate GPU memory for chains
- `burn_in`: Number of initial samples to discard (default: 0)

**Core Functionality**:

#### Simulation Control:
- `generate_samples(progress_bar=True)`: Execute MCMC with performance timing and progress tracking
- `reset()`: Reset simulation state for re-running
- `has_run()`: Check if simulation has been executed (supports pre-allocated chains)

#### Performance Features:
- `benchmark_performance(num_samples_list, compare_cpu=True)`: Comprehensive performance benchmarking
  - Tests multiple sample sizes
  - Compares GPU vs CPU performance
  - Returns detailed timing and speedup metrics
  - Automatic performance reporting

#### Analysis Methods:
- `acceptance_rate()`: Calculate acceptance rate (excluding burn-in)
- `expected_squared_jump_distance()`: GPU-optimized ESJD computation with CPU fallback
- `pt_expected_squared_jump_distance()`: ESJD for parallel tempering

#### Visualization:
- `traceplot(single_dim=False, show=False, use_gpu_data=True)`: GPU-optimized traceplots
  - Optional GPU tensor data usage for faster processing
  - Limits to 5 dimensions for readability
  - Saves to `images/` directory

- `samples_histogram(num_bins=50, axis=0, show=False, use_gpu_data=True)`: GPU-optimized histograms
  - Optional GPU tensor data usage
  - Target density overlay
  - Saves to `images/` with GPU identifier

**Key Design Features**:
- Automatic device detection and management
- Pre-allocation support for memory optimization
- Comprehensive performance benchmarking and reporting
- GPU tensor optimization for visualization
- Backward compatibility with CPU algorithms
- Integrated timing and throughput reporting
- Support for both standard and GPU-accelerated algorithms
- Burn-in handling throughout all computations

This is the primary class for running MCMC simulations with GPU acceleration. Key aspects:

*   **Initialization (`__init__`)**: Takes parameters like `dim`, `proposal_config` (a dictionary specifying the proposal distribution type and its parameters), `num_iterations`, the `algorithm` class (e.g., `RandomWalkMH_GPU_Optimized`), `target_dist`, etc.
    *   It correctly instantiates the specified MCMC algorithm (e.g., `RandomWalkMH_GPU_Optimized`).
    *   Crucially, it uses the `proposal_config` to create the appropriate proposal distribution object (e.g., `NormalProposal`, `LaplaceProposal` from the `proposal_distributions` package) via its internal `_create_proposal_distribution` method. This proposal object is then passed to the MCMC algorithm.
*   **Proposal Handling**: The `_create_proposal_distribution` method is responsible for parsing `proposal_config` and instantiating the correct proposal distribution class from the `proposal_distributions` package (e.g., `NormalProposal`, `LaplaceProposal`, `UniformRadiusProposal`). It imports these classes from `proposal_distributions`.
*   The `interfaces/proposals.py` file has been **removed**. The `ProposalDistribution` base class is now located in `proposal_distributions/base.py`, and concrete implementations are in the `proposal_distributions` directory.

## Architecture Benefits

This interface-based design enables:
1. **Easy Algorithm Comparison**: Swap different MCMC algorithms with identical analysis pipelines
2. **Target Distribution Flexibility**: Test algorithms on various probability distributions with CPU or GPU acceleration
3. **Performance Optimization**: Choose between CPU and GPU implementations based on problem requirements
4. **Consistent Analysis Pipeline**: Standardized metrics and visualization across all experiments
5. **Extensibility**: Add new algorithms or distributions without modifying existing code
6. **Reproducibility**: Built-in seeding and standardized experiment structure
7. **Comprehensive Benchmarking**: Integrated performance testing and comparison tools
8. **Memory Efficiency**: Pre-allocation options and GPU memory management
9. **Burn-in Support**: Automatic handling of burn-in periods across all metrics and analyses