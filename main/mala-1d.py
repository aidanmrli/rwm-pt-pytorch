import numpy as np
import matplotlib.pyplot as plt
import math

def target_distribution(x):
    return np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi)


def gradient(x):
    return -x


def proposal_distribution(x, step_size):
    return np.random.normal(x + (step_size)/2 * gradient(x), np.sqrt(step_size))


def cond(x, y, step_size):
    return math.exp(-((y - x - ((step_size / 2) * gradient(x)))**2 / (2 * step_size)))


def acceptance_probability(current_sample, proposed_sample, step_size):
    numerator = target_distribution(proposed_sample) * cond(current_sample, proposed_sample, step_size)
    denominator = target_distribution(current_sample) * cond(proposed_sample, current_sample, step_size)
    return min(1, numerator / denominator)


def metropolis_adjusted_langevin_algorithm(num_samples, sigma):
    current_sample = np.random.normal(0, 1)
    samples = [current_sample]

    for _ in range(num_samples - 1):
        proposed_sample = proposal_distribution(current_sample, sigma)
        if np.random.rand() < acceptance_probability(current_sample, proposed_sample, sigma):
            current_sample = proposed_sample
        samples.append(current_sample)

    return np.array(samples)

num_samples = 1000
dim = 1
step_size = 0.2

samples_mala = metropolis_adjusted_langevin_algorithm(num_samples, step_size)

plt.figure(figsize=(10, 6))
x = np.linspace(-5, 5, 1000)
plt.plot(x, target_distribution(x), label='Target Distribution', color='red', linewidth=2)
plt.hist(samples_mala, bins=50, density=True, alpha=0.7, label='Generated Samples (MALA)')
plt.title('Metropolis Adjusted Langevin Algorithm (MALA)')
plt.xlabel('Value')
plt.ylabel('Probability Density')


fig, ax = plt.subplots()
ax.plot(samples_mala, color = "skyblue")
_ = ax.set(xlabel = "Samples", ylabel =r'$\mu$')

plt.legend()
plt.show()
