import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import random


def generate_covariance_matrix(dim):
    # variance = random.randint(1,6)
    M = np.random.normal(0, 2, size=(dim, dim))
    Sigma_star = np.dot(M, M.T)
    return Sigma_star


def metropolis_algorithm(target_density, initial_sample, iterations,
                         sigma_squared):
    samples = [initial_sample]
    current_sample = initial_sample
    esjd = 0
    accept = 0
    for _ in range(iterations):
        # Propose a new sample
        # proposal = current_sample + np.random.normal(0, np.sqrt(sigma_squared),
        #                                              size=len(current_sample))
        proposal = current_sample + np.random.multivariate_normal(np.zeros(dim), sigma_squared*Sigma_star)
        # Calculate acceptance ratio
        acceptance_ratio = min(1, target_density(proposal) / target_density(
            current_sample))

        # Accept or reject the proposal
        if np.random.rand() < acceptance_ratio:
            esjd += np.sum(
                np.square(np.array(current_sample) - np.array(proposal)))
            current_sample = proposal
            accept += 1

        samples.append(current_sample)
    acceptance_rate = accept / iterations

    return np.array(samples), esjd, acceptance_rate


# Define the target density
dim = 30
lst_jump = []
acc = []
Sigma_star = generate_covariance_matrix(dim)
target_density = stats.multivariate_normal(mean=np.zeros(dim),
                                           cov=Sigma_star).pdf

# Metropolis parameters
initial_sample = np.zeros(dim)
iterations = 5000
rate = [0.01, 0.05, 0.1, 0.2, 0.4, 0.5, 1, 1.5, 2, 2.5,3,4]
# rate = [0.1888]
# Plot trace plots for each sigma_squared value
for sigma_squared in rate:
    samples, esjd, acceptance_rate = metropolis_algorithm(target_density,
                                                          initial_sample,
                                                          iterations,
                                                          sigma_squared)
    print(acceptance_rate)
    lst_jump.append(esjd/iterations)
    acc.append(acceptance_rate)
    # Trace Plot
    plt.plot(samples[:, 1])
    plt.title(f'Dimension {1} with proposed variance {sigma_squared}')
    plt.xlabel('Iterations')
    plt.ylabel('Sample Value')

    plt.show()

plt.plot(acc, lst_jump)
plt.xlabel('acceptance rate')
plt.ylabel('ESJD')
plt.show()

plt.plot(rate, lst_jump)
plt.xlabel('proposed variance')
plt.ylabel('ESJD')
plt.show()

plt.plot(rate, acc)
plt.xlabel('rate')
plt.ylabel('acc')
plt.show()
