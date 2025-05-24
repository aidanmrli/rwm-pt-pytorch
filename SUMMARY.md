# MCMC Research Codebase - Project Summary

This repository implements a comprehensive framework for studying optimal scaling properties of Markov Chain Monte Carlo (MCMC) algorithms, specifically Random Walk Metropolis (RWM) and Parallel Tempering Random Walk Metropolis (PT-RWM). The codebase is designed to analyze the relationship between acceptance rates and Expected Squared Jumping Distance (ESJD) across various target distributions and dimensionalities.

## 🚀 NEW: GPU Acceleration 

**MAJOR PERFORMANCE IMPROVEMENT**: The codebase now includes GPU-accelerated implementations that provide **10-100x speedup** over the original CPU versions:

- **GPU-Accelerated Algorithms**: `RandomWalkMH_GPU` with PyTorch-based vectorization
- **Smart Memory Management**: Pre-allocated GPU memory for optimal performance
- **Seamless Integration**: Drop-in replacements for existing algorithms with identical interfaces
- **Automatic Fallback**: Works on both GPU and CPU systems automatically

**Performance Example**: 100,000 samples that previously took hours now complete in minutes!

## Project Overview

**Research Focus**: Optimal scaling theory for MCMC algorithms
**Key Metric**: Expected Squared Jumping Distance (ESJD) as a measure of algorithm efficiency
**Primary Algorithms**: Random Walk Metropolis and Parallel Tempering variants
**Target Applications**: High-dimensional sampling, multimodal distributions, algorithm comparison

## Architecture Overview

The codebase follows a modular, interface-based design adhering to SOLID principles:

### Core Components

1. **Interfaces** (`interfaces/`): Abstract base classes defining contracts
   - `TargetDistribution`: Interface for probability distributions
   - `MHAlgorithm`: Interface for Metropolis-Hastings algorithms  
   - `MCMCSimulation`: Simulation controller and analysis framework

2. **Algorithms** (`algorithms/`): MCMC algorithm implementations
   - `RandomWalkMH`: Standard Random Walk Metropolis
   - `ParallelTemperingRWM`: Parallel Tempering with adaptive temperature ladders

3. **Target Distributions** (`target_distributions/`): Test distributions
   - Unimodal: MultivariateNormal, Hypercube, IID products
   - Multimodal: ThreeMixture, RoughCarpet with optional scaling

4. **Experimental Framework**: Scripts for systematic studies
   - `experiment_RWM.py`: Comprehensive RWM parameter sweeps
   - `experiment_pt.py`: Parallel tempering optimization studies

## Key Features

### Algorithm Capabilities
- **Random Walk Metropolis**: Symmetric/asymmetric proposals, temperature scaling
- **Parallel Tempering**: Adaptive temperature ladder construction, swap optimization
- **Numerical Stability**: Log-density calculations, underflow protection
- **Performance Tracking**: Acceptance rates, ESJD computation, convergence diagnostics

### Target Distribution Suite
- **Gaussian Distributions**: Standard multivariate normal with configurable parameters
- **Uniform Distributions**: Hypercube domains with adjustable boundaries
- **IID Products**: Gamma and Beta distribution products for non-Gaussian testing
- **Multimodal Challenges**: Three-component mixtures and rough carpet distributions
- **Scaling Studies**: Optional coordinate-wise scaling for heterogeneous difficulty

### Experimental Infrastructure
- **Parameter Sweeps**: Systematic variance/temperature parameter exploration
- **Multi-seed Averaging**: Statistical robustness through multiple random seeds
- **Automated Data Collection**: JSON storage of performance metrics
- **Visualization Pipeline**: Automated plot generation for analysis

## Research Applications

### Optimal Scaling Studies
- **Acceptance Rate Optimization**: Identify optimal ~0.234 acceptance rate for high dimensions
- **ESJD Maximization**: Find proposal variances that maximize algorithm efficiency
- **Dimensional Scaling**: Study how optimal parameters scale with problem dimension

### Algorithm Comparison
- **RWM vs PT-RWM**: Compare standard and parallel tempering approaches
- **Distribution Difficulty**: Assess algorithm performance across target types
- **Computational Trade-offs**: Analyze efficiency vs computational cost

### Theoretical Validation
- **Optimal Scaling Theory**: Verify theoretical predictions about acceptance rates
- **Temperature Ladder Design**: Optimize parallel tempering temperature schedules
- **Convergence Analysis**: Assess mixing and convergence properties

## Data and Results

### Experimental Data (`data/`)
- **50+ Result Files**: Comprehensive parameter sweeps across distributions and dimensions
- **Standardized Format**: JSON files with ESJD, acceptance rates, and parameter ranges
- **Multi-dimensional Coverage**: 2D to 100D problems for scaling analysis
- **Statistical Robustness**: Multiple seeds for reliable performance estimates

### Visualizations (`images/`)
- **Performance Curves**: ESJD vs acceptance rate relationships
- **Parameter Optimization**: Variance tuning and acceptance rate analysis
- **Diagnostic Plots**: Traceplots and histograms for convergence assessment
- **Publication Figures**: High-resolution plots for research communication

## Usage Patterns

### Single Simulation Analysis
```python
# experiment.py - Individual simulation with diagnostics
simulation = MCMCSimulation(dim=5, sigma=optimal_variance, 
                           algorithm=RandomWalkMH, target_dist=distribution)
chain = simulation.generate_samples()
simulation.traceplot()  # Convergence assessment
simulation.samples_histogram()  # Correctness validation
```

### Systematic Parameter Studies
```bash
# experiment_RWM.py - Comprehensive parameter sweeps
python experiment_RWM.py --dim 20 --target MultivariateNormal --num_seeds 5
# Generates: data files, performance plots, optimal parameter identification
```

### Parallel Tempering Optimization
```bash
# experiment_pt.py - Temperature ladder optimization
python experiment_pt.py --dim 30 --target ThreeMixture --swap_accept_max 0.6
# Generates: adaptive temperature ladders, swap rate analysis
```

## Technical Highlights

### Numerical Stability
- Log-density calculations to prevent underflow in high dimensions
- Numerical constants (1e-300) to avoid log(0) errors
- Robust acceptance probability computation

### Performance Optimization
- **GPU Acceleration**: PyTorch-based GPU-accelerated implementations with 10-100x speedup
- **Memory Management**: Pre-allocated GPU memory for chains to reduce allocation overhead
- Cached target density evaluations to reduce redundant computation
- Efficient state management in parallel tempering
- Vectorized operations where applicable

### GPU-Accelerated Components
- **RandomWalkMH_GPU**: GPU-accelerated Random Walk Metropolis
- **MultivariateNormal_GPU**: GPU-optimized target distribution
- **MCMCSimulation_GPU**: GPU-aware simulation framework with performance benchmarking
- **Automatic Device Detection**: Seamless fallback to CPU when GPU unavailable

### Extensibility
- Plugin architecture for new algorithms and distributions
- GPU/CPU abstraction enables easy performance scaling
- Consistent interfaces enable easy algorithm swapping
- Modular design supports independent component development

## Research Impact

This codebase enables:
1. **Theoretical Validation**: Empirical verification of optimal scaling theory
2. **Algorithm Development**: Framework for testing new MCMC variants
3. **Practical Guidance**: Optimal parameter selection for real applications
4. **Comparative Studies**: Systematic algorithm performance evaluation
5. **Educational Tool**: Clear implementation of fundamental MCMC concepts

The modular design and comprehensive experimental framework make this an ideal platform for MCMC research, algorithm development, and optimal scaling studies in high-dimensional statistical inference. 