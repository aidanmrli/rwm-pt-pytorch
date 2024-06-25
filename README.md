# montecarlo
A modular high-level library for running high-dimensional Metropolis algorithms across a variety of scaling and tempering conditions.
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview
We offer implementations of the following:
- Random-walk Metropolis algorithms with or without parallel tempering. 
- Various target distributions with densities that are used for experimentation. This includes the multivariate normal and various multimodal distributions.
- Visualizations: we have traceplots and histograms for individual simulations, as well as line plots showing the overall trends of ESJD with acceptance rate or proposal variance.
- Experiment suite to run individual simulations, or many simulations with a changed value (such as the scaling), or parallel tempering algorithms.
- Clean architecture adhering to SOLID principles so that this repository can be extended easily with a plug-and-play system for algorithms, target distributions, and proposal distributions in experiments.

## Getting Started
### Installation
To download the repository, navigate to a directory in your terminal and run `https://github.com/aidanmrli/montecarlo.git`.
Once you have this repository locally, navigate into the repository. 
While you are in the root directory, run `pip install -r requirements.txt`. Python comes with pip automatically installed, but if you have any issues please go to `https://pip.pypa.io/en/stable/installation/`.

```bash
git clone https://github.com/aidanmrli/montecarlo.git
cd montecarlo
pip install -r requirements.txt
```

# Experiments

## RWM Experiments

The `experiment_RWM.py` script runs Random Walk Metropolis (RWM) simulations with various parameters.
We admit the following arguments:

- `--dim`: Dimension of the target and proposal distributions (default: 20)
- `--var_max`: Upper bound of the variance value range (default: 2.0)
- `--target`: Target density distribution (default: "Hypercube")
- `--num_iters`: Number of iterations for each MCMC simulation (default: 100000)
- `--init_seed`: Starting seed value (default: 0)
- `--num_seeds`: Number of seeds to use in the simulations (default: 5)

Sample usage with custom parameters:
``python experiment_RWM.py --dim 10 --var_max 3.0 --target MultivariateNormal --num_iters 200000 --init_seed 42 --num_seeds 5``

The script generates:

1. JSON data file with simulation results in the `data/` directory
2. Three plots saved in the `images/` directory:
   - ESJD vs acceptance rate
   - Acceptance rate vs variance value
   - ESJD vs variance value

The script also prints the maximum ESJD, corresponding acceptance rate, and variance value to the console.

### Available Target Distributions

- MultivariateNormal
- RoughCarpet
- RoughCarpetScaled
- ThreeMixture
- ThreeMixtureScaled
- Hypercube
- IIDGamma
- IIDBeta

## PT Experiments

The `experiment_pt.py` script runs Parallel Tempering (PT) simulations with various parameters. In our implementation, we propose a swap between chains every 20 iterations.
We admit the following arguments:
parser.add_argument("--swap_accept_max", type=float, default=0.6, help="Upper bound of the swap acceptance rate range")
    parser.add_argument("--target", type=str, default="RoughCarpet", help="Target density distribution")
    parser.add_argument("--num_iters", type=int, default=100000, help="Number of iterations for the MCMC simulation")
    parser.add_argument("--init_seed", type=int, default=0, help="Starting seed value")
    parser.add_argument("--num_seeds", type=int, default=5, help="Number of seeds to use in the simulations")

- `--dim`: Dimension of the target and proposal distributions (default: 20)
- `--swap_accept_max`: Upper bound of the swap acceptance rate range (default: 0.6)
- `--target`: Target density distribution (default: "RoughCarpet")
- `--num_iters`: Number of iterations for each MCMC simulation (default: 100000)
- `--init_seed`: Starting seed value (default: 0)
- `--num_seeds`: Number of seeds to use in the simulations (default: 5)

Sample usage with custom parameters:
``python experiment_RWM.py --dim 10 --var_max 3.0 --target MultivariateNormal --num_iters 200000 --init_seed 42 --num_seeds 5``

The script generates:

1. JSON data file with simulation results in the `data/` directory
2. Two plots saved in the `images/` directory:
   - ESJD vs acceptance rate (constructed swap rates, we use this for our results)
   - ESJD vs acceptance rate (actual swap rates in finite time)

### Available Target Distributions

- MultivariateNormal
- RoughCarpet
- ThreeMixture

## Running a single MCMC simulation and produce a histogram and traceplot:
Run experiment.py (in the root directory). Adjust the simulation arguments in experiment.py as necessary before the experiment.

The MCMCSimulation class can generate visualizations such as a histogram in a single dimension, and the traceplot of the Markov chain in a single dimension (or all dimensions).

## Plotting
The `plot.py` script generates the ESJD vs acceptance rate plots used for the article. It makes a plot for each JSON data file in the `data/` directory and stores it in the `images/publishing/` directory.

Simply run the plot.py file, no arguments are needed:
``python plot.py``

## Directory Structure
General classes with their attributes and methods are in the interfaces folder. 
- MCMCSimulation is a class for running a single MCMC simulation for generating samples from a target distribution and visualizing the various metrics and results.
- TargetDistribution is a general interface for specifying the methods of a target distribution. The implementations of this class in the target_distributions folder implement these methods, such as the density.
- MHAlgorithm is a general interface for a Metropolis-Hastings algorithm for sampling for a target distribution.  The implementations of this class in the algorithms folder implement these methods, such as calculating the (log) acceptance probability.

Algorithm implementations are in the algorithms folder. Likewise, target distribution implementations are in the target_distributions folder. These implementations _inherit_ the methods and attributes defined by their parents, allowing for a greater degree of consistency between different implementations.

## License
This repository is MIT licensed. See the LICENSE file in the root directory.
