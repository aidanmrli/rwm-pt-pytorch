# Images Directory

This directory `images/` contains visualization outputs from MCMC algorithm performance studies. The plots provide graphical analysis of the relationship between algorithm parameters (proposal variance, acceptance rates) and performance metrics (ESJD), enabling visual assessment of optimal scaling properties.

## Overview

The images directory serves as the repository for:
- **Performance Curves**: ESJD vs variance and acceptance rate relationships
- **Algorithm Diagnostics**: Traceplots and sample histograms for convergence assessment
- **Comparative Analysis**: Visual comparison across distributions, algorithms, and dimensions
- **Publication Figures**: High-resolution plots suitable for research publications

## File Naming Convention

All image files follow a standardized naming pattern:
```
{PlotType}_{DistributionName}_{AlgorithmName}_dim{Dimension}_seed{Seed}_{Iterations}iters.png
```

**Components**:
- **PlotType**: Type of visualization (see Plot Types section)
- **DistributionName**: Target distribution identifier
- **AlgorithmName**: MCMC algorithm used (`RWM` or `PTrwm`)
- **Dimension**: Problem dimensionality
- **Seed**: Random seed (when specified)
- **Iterations**: Number of MCMC steps

## Plot Types

### Performance Analysis Plots

#### `ESJDvsVar_*`
**Purpose**: Expected Squared Jumping Distance vs Proposal Variance
**Content**: 
- X-axis: Proposal variance (log scale, typically 1e-6 to 8.0)
- Y-axis: ESJD values
- Shows how algorithm efficiency varies with proposal variance
- Identifies optimal variance for maximum ESJD

#### `AcceptvsVar_*`
**Purpose**: Acceptance Rate vs Proposal Variance
**Content**:
- X-axis: Proposal variance (log scale)
- Y-axis: Acceptance rate (0 to 1)
- Shows monotonic decrease in acceptance rate with increasing variance
- Complements ESJD analysis for parameter tuning

#### `ESJDvsAccept_*`
**Purpose**: Expected Squared Jumping Distance vs Acceptance Rate
**Content**:
- X-axis: Acceptance rate
- Y-axis: ESJD values
- **Key Plot**: Reveals optimal acceptance rate for maximum efficiency
- Validates theoretical predictions (optimal ≈ 0.234 for high dimensions)

### Parallel Tempering Specific Plots

#### `ESJDvsSwapAcceptActual_*`
**Purpose**: PT-ESJD vs Actual Swap Acceptance Rate
**Content**:
- X-axis: Measured swap acceptance rate during simulation
- Y-axis: Parallel tempering ESJD
- Shows relationship between temperature exchange efficiency and overall performance

#### `ESJDvsSwapAcceptConstr_*`
**Purpose**: PT-ESJD vs Constructed Swap Acceptance Rate
**Content**:
- X-axis: Target swap acceptance rate used in temperature ladder construction
- Y-axis: Parallel tempering ESJD
- Evaluates effectiveness of adaptive temperature ladder construction

### Diagnostic Plots

#### `traceplot_*`
**Purpose**: Markov Chain Traceplots
**Content**:
- X-axis: MCMC iteration number
- Y-axis: Parameter values across dimensions
- Multiple lines for multi-dimensional problems
- Assesses chain mixing and convergence

#### `hist_*`
**Purpose**: Sample Histograms with Target Density Overlay
**Content**:
- Histogram: Empirical distribution of MCMC samples
- Red dashed line: True target density
- Validates correctness of sampling (convergence to target)
- Focuses on first coordinate for visualization

## Directory Structure

### Main Directory (`images/`)
Contains the bulk of experimental visualizations organized by:
- Distribution type (MultivariateNormal, ThreeMixture, IIDGamma, etc.)
- Algorithm type (RWM vs PTrwm)
- Dimensionality (2D to 100D)
- Plot type (performance curves, diagnostics)

### `publishing/` Subdirectory
**Purpose**: High-resolution, publication-ready figures
**Content**: Curated subset of main plots with enhanced formatting
**Format**: 300 DPI PNG files suitable for academic publications

## Coverage Analysis

### Distributions Visualized
- **Unimodal**: MultivariateNormal, Hypercube, IIDGamma, IIDBeta
- **Multimodal**: ThreeMixture, ThreeMixtureScaled, RoughCarpet, RoughCarpetScaled

### Algorithms Analyzed
- **RWM**: Comprehensive coverage across all distributions and dimensions
- **PTrwm**: Focus on multimodal distributions (ThreeMixture, RoughCarpet, MultivariateNormal)

### Dimensional Scaling
- **Low Dimension**: 2D, 5D for detailed analysis
- **Medium Dimension**: 10D, 20D for scaling studies
- **High Dimension**: 30D, 50D, 100D for asymptotic behavior

## Key Insights from Visualizations

### Optimal Scaling Relationships
1. **Acceptance Rate**: Optimal ≈ 0.234 for high-dimensional problems
2. **ESJD Curves**: Characteristic inverted-U shape with clear maximum
3. **Dimensional Scaling**: Optimal variance scales with dimension

### Algorithm Comparison
1. **RWM Performance**: Effective for unimodal distributions
2. **PT-RWM Advantage**: Superior performance on multimodal targets
3. **Computational Trade-off**: PT-RWM higher cost but better exploration

### Distribution Difficulty
1. **Easiest**: MultivariateNormal, Hypercube (smooth, unimodal)
2. **Moderate**: IIDGamma, IIDBeta (product structure)
3. **Challenging**: ThreeMixture, RoughCarpet (multimodal, mode separation)

## File Statistics

- **Total Images**: ~100+ visualization files
- **File Sizes**: 17KB - 54KB per PNG file
- **Resolution**: High-resolution suitable for publication
- **Coverage**: Complete experimental parameter space
- **Format**: PNG with transparent backgrounds where applicable

## Usage Applications

The visualizations enable:
1. **Parameter Optimization**: Identify optimal proposal variances and acceptance rates
2. **Algorithm Validation**: Verify theoretical predictions about optimal scaling
3. **Performance Comparison**: Compare algorithm efficiency across problem settings
4. **Research Communication**: Publication-ready figures for academic papers
5. **Diagnostic Analysis**: Assess convergence and correctness of MCMC sampling 