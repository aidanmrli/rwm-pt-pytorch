import numpy as np
import pymc as pm
import matplotlib.pyplot as plt

# Define the dimensionality of the density
ndim = 1

# Generate some synthetic data
true_mean = np.zeros(ndim)
true_cov = np.eye(ndim)
observed_data = np.random.multivariate_normal(true_mean, true_cov, size=100)

# Define the PyMC model
basic_model = pm.Model()
with basic_model:
    # Define the priors for the parameters
    print('ndim:', ndim)
    mu = pm.Normal('mu', mu=0, sigma=1, shape=ndim)
    sigma = pm.HalfNormal('sigma', sigma=1, shape=ndim)
    
    # Define the likelihood (assuming normally distributed data)
    print('mu:', mu)
    likelihood = pm.Normal('likelihood', mu=mu, sigma=sigma, observed=observed_data)
    
    # Define the Metropolis sampler
    step = pm.Metropolis()
    
    # Perform sampling
    trace = pm.sample(1000, step=step, random_seed=42)

# Print acceptance rate
acceptance_rate = np.mean(trace['sigma'][1:] != trace['sigma'][:-1])
print("Acceptance Rate: {:.2%}".format(acceptance_rate))

# Visualize the chain
pm.traceplot(trace)
plt.show()
