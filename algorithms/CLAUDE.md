# Algorithms Directory

This directory `algorithms/` contains the core MCMC algorithm implementations, specifically Random Walk Metropolis (RWM) and Parallel Tempering Random Walk Metropolis (PT-RWM) algorithms. These implementations inherit from the `MHAlgorithm` interface and provide the actual sampling logic for exploring target distributions.

## Overview

The algorithms implement two main approaches to MCMC sampling:
1. **Standard Random Walk Metropolis**: Basic MCMC for unimodal or well-connected distributions
2. **Parallel Tempering**: Enhanced MCMC for multimodal distributions using temperature scheduling

Both algorithms are designed to study optimal scaling properties and the relationship between acceptance rates and expected square jumping distances (ESJD).

## Files Documentation

### `__init__.py`
**Purpose**: Module initialization file that exports all algorithm classes.
**Exports**:
- All classes from `rwm` module (Random Walk Metropolis)
- All classes from `pt_rwm` module (Parallel Tempering Random Walk Metropolis)

### `rwm.py`
**Purpose**: Implements the standard Random Walk Metropolis-Hastings algorithm.
**Classes**:
- `RandomWalkMH`: Core RWM implementation for sampling from target distributions

**Key Features**:
- **Symmetric Proposals**: Uses multivariate normal proposals centered at current state
- **Temperature Support**: Includes `beta` parameter for integration with parallel tempering
- **Numerical Stability**: Uses log-density calculations to avoid underflow
- **Adaptive Proposal Variance**: Proposal covariance scales as `(var/beta) * I`

**Core Algorithm Logic**:
1. Propose new state: `x' ~ N(x_current, (σ²/β) * I)`
2. Calculate acceptance probability: `min(1, exp(β * [log π(x') - log π(x)]))`
3. Accept/reject proposal using Metropolis criterion
4. Track acceptance statistics for analysis

**Methods**:
- `step()`: Execute one MCMC iteration
- `log_accept_prob()`: Compute log acceptance probability with numerical stability
- `get_name()`: Returns "RWM" identifier

**Implementation Details**:
- Supports both symmetric and asymmetric proposals (though asymmetric case includes proposal density ratio)
- Caches log target density of current state to avoid redundant computation
- Uses small numerical constant (1e-300) to prevent log(0) errors
- Updates acceptance rate incrementally as chain progresses

### `pt_rwm.py`
**Purpose**: Implements Parallel Tempering with Random Walk Metropolis for multimodal sampling.
**Classes**:
- `ParallelTemperingRWM`: Advanced MCMC using multiple temperature levels

**Key Architecture**:
- **Multiple Chains**: Maintains array of `RandomWalkMH` instances, one per temperature
- **Temperature Ladder**: Sequence of inverse temperatures β₁ = 1 > β₂ > ... > βₖ > 0
- **State Swapping**: Periodic attempts to exchange states between adjacent temperature chains
- **Cold Chain Target**: Primary samples come from β = 1 (original target distribution)

**Temperature Ladder Construction**:
1. **Manual Specification**: Use provided `beta_ladder` parameter
2. **Geometric Spacing**: Construct ladder with geometric progression (β_{i+1} = c * β_i)
3. **Adaptive Construction**: Iteratively build ladder targeting specific swap acceptance rate

**Adaptive Ladder Algorithm**:
- Target swap acceptance rate (default: 0.234)
- Use simulation-based approach to estimate swap probabilities
- Employ stochastic approximation: ρₙ₊₁ = ρₙ + (p̂ₙ - p_target) / n^0.25
- Continue until βₘᵢₙ threshold reached

**State Swapping Mechanism**:
- Attempt swaps every `swap_every` steps (default: 20)
- Adjacent chain swaps only: (i, i+1) pairs
- Metropolis acceptance: `min(1, exp((βⱼ - βₖ)(log π(xₖ) - log π(xⱼ))))`
- Track swap statistics for ESJD calculation

**Key Methods**:
- `step()`: Advance all chains and attempt swaps according to schedule
- `attempt_swap(j, k)`: Execute swap attempt between chains j and k
- `log_swap_prob(j, k)`: Calculate log swap acceptance probability
- `construct_beta_ladder_iteratively()`: Build adaptive temperature schedule

**Performance Metrics**:
- **Swap Acceptance Rate**: Fraction of successful state exchanges
- **PT-ESJD**: Expected squared jumping distance in temperature space
- **Individual Chain Statistics**: Each RWM chain tracks its own acceptance rate

**Implementation Highlights**:
- Efficient state management with proper copying to avoid reference issues
- Caching of log target densities across temperature levels
- Support for different ladder construction strategies
- Integration with base `MHAlgorithm` interface for consistent analysis

## Algorithm Comparison

| Feature | RWM | PT-RWM |
|---------|-----|--------|
| **Target Use Case** | Unimodal distributions | Multimodal distributions |
| **Computational Cost** | O(d) per step | O(k×d) per step (k = # temperatures) |
| **Mode Exploration** | Local exploration | Global exploration via temperature |
| **Parameter Tuning** | Proposal variance only | Variance + temperature ladder |
| **Convergence** | May get stuck in modes | Enhanced mixing between modes |
| **Analysis Focus** | Acceptance rate vs ESJD | Swap rate vs PT-ESJD |

## Optimal Scaling Theory

Both algorithms are designed to investigate optimal scaling properties:
- **RWM**: Optimal acceptance rate ≈ 0.234 for high dimensions
- **PT-RWM**: Optimal swap acceptance rate ≈ 0.234 for temperature exchanges
- **ESJD Maximization**: Key metric for evaluating algorithm efficiency
- **Dimension Scaling**: Study how optimal parameters scale with problem dimension