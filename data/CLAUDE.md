# Data Directory

This directory `data/` contains experimental results from MCMC algorithm performance studies. The data files store comprehensive metrics for analyzing the relationship between acceptance rates and expected squared jumping distances (ESJD) across different target distributions, algorithms, and dimensionalities.

## Overview

The data directory serves as the repository for:
- **Performance Metrics**: ESJD and acceptance rate measurements across parameter sweeps
- **Algorithm Comparisons**: Results from both RWM and PT-RWM algorithms
- **Scaling Studies**: Multi-dimensional analysis from 2D to 100D problems
- **Distribution Testing**: Results across various target distributions (Gaussian, multimodal, IID products, etc.)
- **Seed Averaging**: Smoothed results by averaging across multiple random seeds

## File Naming Convention

All data files follow a standardized naming pattern:
```
{DistributionName}_{AlgorithmName}_dim{Dimension}_seed{Seed}_{Iterations}iters.json
```

**Components**:
- **DistributionName**: Target distribution identifier (e.g., `MultivariateNormal`, `ThreeMixture`, `IIDGamma`)
- **AlgorithmName**: MCMC algorithm used (`RWM` or `PTrwm`)
- **Dimension**: Problem dimensionality (2, 5, 10, 20, 30, 50, 100)
- **Seed**: Random seed for reproducibility (when specified)
- **Iterations**: Number of MCMC steps (typically 100,000)

**Averaged Files**: Files ending with `_averaged.json` contain results averaged across multiple seeds for smoother plotting curves.

## Data Structure

Each JSON file contains the following standardized fields:

### Core Metrics
- **`max_esjd`**: Maximum expected squared jumping distance achieved
- **`max_acceptance_rate`**: Acceptance rate corresponding to maximum ESJD
- **`max_variance_value`**: Proposal variance that yielded maximum ESJD

### Performance Arrays
- **`expected_squared_jump_distances`**: ESJD values across variance parameter sweep
- **`acceptance_rates`**: Corresponding acceptance rates for each variance value
- **`var_value_range`**: Proposal variance values tested (typically 40 points from 1e-6 to 8.0)

### Averaged File Additional Fields
- **`averaged_from_seeds`**: List of random seeds that were averaged
- **`num_files_averaged`**: Number of individual seed files combined
- **`source_files`**: Names of the original files that were averaged

### Data Interpretation
- Arrays are aligned by index: `acceptance_rates[i]` and `expected_squared_jump_distances[i]` correspond to `var_value_range[i]`
- Used for plotting acceptance rate vs ESJD curves and identifying optimal scaling relationships
- Enables comparison of algorithm efficiency across different problem settings

## Target Distributions Tested

### Unimodal Distributions
- **`MultivariateNormal`**: Standard Gaussian with identity covariance
- **`Hypercube`**: Uniform distribution over [0,1]^d
- **`IIDGamma`**: Product of independent Gamma(2,3) distributions
- **`IIDBeta`**: Product of independent Beta(2,3) distributions

### Multimodal Distributions
- **`ThreeMixture`**: Three-component Gaussian mixture with modes at (-15,0,...,0), (0,0,...,0), (15,0,...,0)
- **`ThreeMixtureScaled`**: Three-mixture with random coordinate-wise scaling
- **`RoughCarpet`**: Product of 1D three-mode distributions with unequal weights [0.5, 0.3, 0.2]
- **`RoughCarpetScaled`**: Rough carpet with random coordinate-wise scaling

## Algorithm Coverage

### Random Walk Metropolis (RWM)
- **File Pattern**: `*_RWM_*`
- **Focus**: Standard MCMC performance on various distributions
- **Dimensions Tested**: 2, 5, 10, 20, 30, 50, 100
- **Key Insight**: Optimal acceptance rate ≈ 0.234 for high dimensions

### Parallel Tempering RWM (PT-RWM)
- **File Pattern**: `*_PTrwm_*`
- **Focus**: Enhanced sampling for multimodal distributions
- **Dimensions Tested**: 20, 30 (primarily for multimodal targets)
- **Key Insight**: Swap acceptance rate optimization for temperature ladder efficiency

## Files Documentation

### `average_seeds.py` ⭐ NEW
**Purpose**: Average MCMC experimental results across multiple random seeds to create smoother plotting curves.

**Functionality**:
- Finds all JSON files matching a specific experimental configuration
- Groups files by target distribution, algorithm, dimension, and iterations
- Averages ESJD and acceptance rate values element-wise across seeds
- Creates output files with `_averaged.json` suffix for smoother plotting

**Key Features**:
- **Pattern Matching**: Automatically finds files with same experimental setup but different seeds
- **Element-wise Averaging**: Averages corresponding array elements across all seed files
- **Metadata Preservation**: Maintains variance ranges and experimental parameters
- **Documentation**: Tracks which seeds were averaged and source files used

**Usage Examples**:
```bash
# Average specific experimental configuration
python average_seeds.py --target MultivariateNormal --algorithm RWM_GPU --dim 20 --iters 1000

# Average using pattern directly
python average_seeds.py --pattern "MultivariateNormal_RWM_GPU_dim20_1000iters"
```

**Output**: Creates files like `MultivariateNormal_RWM_GPU_dim20_1000iters_seeds4_7_9_10_averaged.json`

### `batch_average_seeds.py` ⭐ NEW
**Purpose**: Batch processing script that automatically finds and averages all experimental configurations with multiple seeds.

**Functionality**:
- Scans data directory for all experimental patterns
- Identifies configurations with multiple seed files (≥2 by default)
- Automatically creates averaged versions for smoother plotting
- Provides dry-run capability to preview operations

**Key Features**:
- **Automatic Discovery**: Finds all multi-seed experimental configurations
- **Batch Processing**: Processes all eligible patterns in one command
- **Safety Features**: Dry-run mode and existing file detection
- **Flexible Filtering**: Configurable minimum seed requirements

**Usage Examples**:
```bash
# Process all multi-seed groups in current directory
python batch_average_seeds.py

# Dry run to see what would be processed
python batch_average_seeds.py --dry_run --verbose

# Require minimum 3 seeds for averaging
python batch_average_seeds.py --min_seeds 3
```

**Benefits**: 
- **Smoother Curves**: Reduces statistical noise from individual random seeds
- **Better Visualization**: Creates cleaner plots for publication and analysis
- **Automated Workflow**: No manual identification of seed groups required

### `combine_data.py`
**Purpose**: Legacy utility script for averaging results from exactly two random seeds.
**Functionality**:
- Loads two JSON files with identical structure
- Averages all numeric fields (scalars and arrays element-wise)
- Preserves variance range (assumed identical across seeds)
- Outputs combined results to new JSON file

**Key Functions**:
- `load_json(file_path)`: Load experimental data from JSON
- `average_lists(list1, list2)`: Element-wise averaging of performance arrays
- `combine_json(file1, file2, output_file)`: Complete data combination workflow

**Usage Example**: Combines results from different random seeds to reduce statistical noise in performance measurements.

**Note**: For new work, use `average_seeds.py` or `batch_average_seeds.py` which support any number of seeds and provide better functionality.

## Data Analysis Applications

The stored data enables several types of analysis:

1. **Optimal Scaling Studies**: Plot acceptance rate vs ESJD to identify optimal proposal variances
2. **Dimensional Scaling**: Compare how optimal parameters change with problem dimension
3. **Algorithm Comparison**: Evaluate RWM vs PT-RWM performance on multimodal problems
4. **Distribution Difficulty**: Assess which target distributions are most challenging for MCMC
5. **Reproducibility**: Multiple seeds allow for statistical significance testing
6. **Smooth Visualization**: Averaged files provide cleaner curves for publication-quality plots

## Workflow for Smooth Plotting

### Recommended Process:
1. **Generate Individual Seed Data**: Run experiments with multiple random seeds
2. **Create Averaged Data**: Use `batch_average_seeds.py` to automatically average across seeds
3. **Plot Results**: Use `plot.py` with averaged files for smoother curves
4. **Compare Configurations**: Analyze both individual and averaged results

### Example Workflow:
```bash
# 1. Generate data with multiple seeds (done via experiment scripts)
# Files: MultivariateNormal_RWM_GPU_dim20_1000iters_seed{4,7,9,10}.json

# 2. Create averaged versions
cd data/
python batch_average_seeds.py --verbose

# 3. Generate plots (including averaged data)
cd ..
python plot.py

# Result: Smoother ESJD vs acceptance rate curves for analysis
```

## File Statistics

- **Total Files**: ~50+ experimental result files
- **Size Range**: 2.2KB - 3.6KB per file
- **Coverage**: 8 target distributions × 2 algorithms × multiple dimensions
- **Iterations**: Consistent 100,000 MCMC steps per experiment (some 1,000 for testing)
- **Parameter Sweep**: 40 variance values per experiment for detailed performance curves
- **Seed Averaging**: Automated averaging across multiple seeds for smoother analysis