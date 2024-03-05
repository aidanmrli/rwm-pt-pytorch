# montecarlo
A modular high-level library to run high-dimensional Monte Carlo algorithms (e.g. Metropolis algorithm) simulations across a variety of scaling and annealing conditions.

## To run a single experiment:
Run experiment.py (in the root directory). Adjust the simulation arguments as necessary before the experiment: 
- change the dimension of the target and proposal distributions
- proposal variance (sigma)
- number of iterations
- choice of algorithm
- target distribution
- symmetry of the proposal distribution (Metropolis vs Metropolis-Hastings)
- numpy seed

The MCMCSimulation class can generate visualizations such as a histogram in a single dimension, and the traceplot of the Markov chain in a single dimension (or all dimensions).

## Directory Structure
General classes with their attributes and methods are in the main folder. 
- MCMCSimulation is a class for running a single MCMC simulation for generating samples from a target distribution and visualizing the various metrics and results.
- TargetDistribution is a general interface for specifying the methods of a target distribution. The implementations of this class in the target_distributions folder implement these methods, such as the density.
- MHAlgorithm is a general interface for a Metropolis-Hastings algorithm for sampling for a target distribution.  The implementations of this class in the algorithms folder implement these methods, such as calculating the (log) acceptance probability.

Algorithm implementations are in the algorithms folder. Likewise, target distribution implementations are in the target_distributions folder.

pymc is a folder for experimenting with PyMC. As of Mar 5, 2024, we have not done any meaningful work with PyMC.
