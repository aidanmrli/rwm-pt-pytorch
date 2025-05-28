# Random Walk Metropolis Algorithms
A high-performance, GPU-accelerated library for running high-dimensional Random Walk Metropolis algorithms across a variety of scaling and tempering conditions. Written completely in Python.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

This repository provides a comprehensive framework for studying optimal scaling properties of Markov Chain Monte Carlo (MCMC) algorithms, specifically Random Walk Metropolis (RWM) and Parallel Tempering Random Walk Metropolis (PT-RWM). The codebase features GPU acceleration with 10-50x performance improvements.

### Key Features

- **GPU Acceleration**: PyTorch-based GPU implementations with JIT compilation and kernel fusion
- **RWM Algorithms in Python**: Random Walk Metropolis with and without parallel tempering
- **Diverse Target Distributions**: Unimodal and multimodal test distributions for algorithm evaluation
- **Rigorous Testing Framework**: Extensive correctness and performance validation
- **Publication-Ready Visualizations**: Automated plot generation for research communication
- **Modular Architecture**: Clean, extensible design following SOLID principles

## Quick Start

### Installation

Run the following in your terminal:

```bash
git clone https://github.com/aidanmrli/random-walk-metropolis.git
cd random-walk-metropolis
pip install -r requirements.txt
```

For maximum performance, install PyTorch with GPU support. On a Linux system, running the standard command as per the [PyTorch website](https://pytorch.org/get-started/locally/) should be fine:

```bash
# For CUDA 12.6
pip install torch torchvision torchaudio
```

### Quick Test

Verify GPU acceleration is working:

```bash
python tests/quick_test.py
```

## Usage Examples

### GPU-Accelerated Experiments

```python
from interfaces import MCMCSimulation_GPU
from algorithms import RandomWalkMH_GPU
from target_distributions import MultivariateNormalTorch

# GPU-accelerated simulation
simulation = MCMCSimulation_GPU(
    dim=20,
    sigma=0.1,
    num_iterations=100000,
    algorithm=RandomWalkMH_GPU,
    target_dist=MultivariateNormalTorch(20),
    pre_allocate=True  # Enable memory optimization
)

chain = simulation.generate_samples()
```

## Experimental Scripts

### GPU-Accelerated RWM Experiments

The `experiment_RWM_GPU.py` script provides comprehensive GPU-accelerated parameter sweeps:

```bash
python experiment_RWM_GPU.py --dim 20 --var_max 4.0 --target MultivariateNormal --num_iters 100000
```

**Arguments:**
- `--dim`: Dimension of the target distribution (default: 20)
- `--target`: Target distribution name (default: "MultivariateNormal")
- `--num_iters`: Number of MCMC iterations (default: 100000)
- `--mode`: Experiment mode - "comparison", "optimization", or "benchmark"
- `--var_max`: The maximum variance is calculated from this number by doing (var ** 2 / dim).
- `--seed`: Random seed for reproducibility

### Traditional CPU Experiments

Original CPU implementations are still available:

```bash
# RWM parameter sweeps
python experiment_RWM.py --dim 10 --var_max 3.0 --target MultivariateNormal --num_iters 200000

# Parallel tempering experiments  
python experiment_pt.py --dim 20 --swap_accept_max 0.6 --target RoughCarpet --num_iters 100000
```

## Available Target Distributions

### GPU-Optimized Distributions
- `MultivariateNormalTorch`: Standard multivariate Gaussian
- `RoughCarpetDistributionTorch`: Multimodal rough landscape
- `ThreeMixtureDistributionTorch`: Three-component Gaussian mixture
- `HypercubeTorch`: Uniform distribution on hypercube
- `IIDGammaTorch`: Product of independent Gamma distributions
- `IIDBetaTorch`: Product of independent Beta distributions

### CPU Distributions
- `MultivariateNormal`, `RoughCarpet`, `ThreeMixture`, `Hypercube`, `IIDGamma`, `IIDBeta`

## Data Processing and Analysis

### Automated Seed Averaging

Reduce statistical noise by averaging across multiple random seeds:

```bash
# Average results for specific configuration
python data/average_seeds.py --target MultivariateNormal --algorithm RWM_GPU --dim 20 --iters 100000

# Batch process all configurations that have multiple seeds
python data/batch_average_seeds.py --base_dir data/ --output_dir averaged_results/
```

### Visualization

Generate publication-ready plots:

```bash
# Create plots for all data files
python plot.py

# Individual simulation diagnostics
python experiment.py  # Generates traceplots and histograms
```

## Testing and Validation

### Comprehensive Test Suite

```bash
# Quick functionality verification
python tests/quick_test.py

# Detailed correctness testing
python tests/test_rwm_correctness.py

# Performance benchmarking
python tests/test_rwm_performance_benchmark.py
```

### Test Coverage
- **Correctness verification**: GPU implementations match CPU within 5%
- **Statistical validation**: Empirical properties match theoretical expectations
- **Performance benchmarking**: Speedup measurement across problem sizes
- **Sequential dependence**: Proper MCMC autocorrelation structure

## Architecture

### Core Components

1. **Interfaces** (`interfaces/`): Abstract base classes
   - `MCMCSimulation` / `MCMCSimulation_GPU`: Simulation controllers
   - `TargetDistribution`: Distribution interface
   - `MHAlgorithm`: Metropolis-Hastings algorithm interface

2. **Algorithms** (`algorithms/`): MCMC implementations
   - `RandomWalkMH` / `RandomWalkMH_GPU`: Standard and GPU-accelerated RWM
   - `RandomWalkMH_GPU_Optimized`: Ultra-fused GPU implementation
   - `ParallelTemperingRWM`: Parallel tempering with adaptive ladders

3. **Target Distributions** (`target_distributions/`): Test distributions
   - CPU versions: Standard NumPy implementations
   - GPU versions: PyTorch tensor implementations with `Torch` suffix

4. **Testing** (`tests/`): Comprehensive validation suite
   - Correctness verification, performance benchmarking, device compatibility

5. **Data Processing** (`data/`): Analysis utilities
   - Seed averaging, batch processing, result aggregation

## Advanced Features

### GPU Optimization Techniques
- **JIT Compilation**: PyTorch JIT for maximum arithmetic efficiency
- **Kernel Fusion**: Complete MCMC steps in single GPU kernels
- **Memory Pre-allocation**: Eliminates dynamic GPU memory management
- **Sequential Processing**: Maintains RWM theoretical correctness

### Research Applications
- **Optimal Scaling Studies**: Identify optimal acceptance rates (~0.234)
- **ESJD Maximization**: Find proposal variances maximizing efficiency
- **Algorithm Comparison**: Compare RWM vs PT-RWM performance
- **Dimensional Scaling**: Study parameter scaling with problem dimension

## Documentation

- **`GPU_QUICKSTART.md`**: Complete guide for GPU acceleration
- **`SUMMARY.md`**: Comprehensive project overview
- **Inline documentation**: Extensive docstrings optimized for AI assistants

## HPC Integration

For high-performance computing environments:

```bash
# SLURM job submission
sbatch run_rwm_gpu_dcs.sbatch

# Manual GPU allocation
srun -p ml -q ml -A ml -w concerto1 -c 4 --mem=2G --gres=gpu:1 --pty bash
```

## Migration from CPU to GPU

### Simple Replacement
```python
# Before (CPU)
from interfaces import MCMCSimulation
from algorithms import RandomWalkMH
from target_distributions import MultivariateNormal

# After (GPU)
from interfaces import MCMCSimulation_GPU
from algorithms import RandomWalkMH_GPU
from target_distributions import MultivariateNormalTorch
```

### Performance Optimization
- Add `pre_allocate=True` for memory optimization
- Use GPU-native target distributions (with `Torch` suffix)
- Enable progress bars with `progress_bar=True`

## Contributing

This repository is designed for easy extension:
- **Modular architecture**: Add new algorithms by inheriting from `MHAlgorithm`
- **Plugin system**: New target distributions inherit from `TargetDistribution`
- **Comprehensive testing**: All new features should include correctness tests

## License

This repository is MIT licensed. See the LICENSE file for details.