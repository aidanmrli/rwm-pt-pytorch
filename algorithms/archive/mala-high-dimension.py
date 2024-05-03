import numpy as np
import matplotlib.pyplot as plt


def target_distribution(x, dim):
    product = 1
    for i in range(dim):
        value = np.exp(-0.5 * x[i] ** 2) / np.sqrt(2 * np.pi)
        product *= value
    return np.prod(product)


def grad_log_target(x):
    return -x


def proposal_distribution(x, step_size):
    return np.random.normal(x + 0.5 * step_size * grad_log_target(x), step_size)


def q(x, y, dim, step_size):
    product = 1
    for i in range(dim):
        value = np.exp((-(y[i]-x[i]-step_size*(-x)/2)**2)/2*step_size)/np.sqrt(2 * np.pi*step_size)
        product *= value
    return np.prod(product)


def acceptance_probability(current_sample, proposed_sample, step_size):
    numerator = target_distribution(proposed_sample, dim) * q(proposed_sample, current_sample, dim, step_size)
    denominator = target_distribution(current_sample, dim) * q(current_sample, proposed_sample, dim, step_size)
    return min(1, numerator / denominator)


def mala_sampler(dim, num_samples, step_size):
    samples = [np.zeros(dim)]
    x_initial = np.zeros(dim)
    esjd = 0
    for i in range(num_samples - 1):
        proposal = proposal_distribution(x_initial, step_size)

        acceptance_ratio = acceptance_probability(x_initial, proposal, step_size)

        if np.random.rand() < acceptance_ratio:
            esjd += np.sum(np.square(np.array(x_initial) - np.array(proposal)))
            x_initial = proposal
        samples.append(x_initial)

    return np.array(samples), esjd

dim = 3
num_samples = 1000
step_size = 0.2

samples, esjd = mala_sampler(dim, num_samples, step_size)

print("estimated jumping square",esjd)

#trace plot
fig, ax = plt.subplots()
ax.plot(samples)
_ = ax.set(xlabel="Samples", ylabel=r'$\mu$')

plt.legend()
plt.show()
