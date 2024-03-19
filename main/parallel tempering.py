# import numpy as np
# import matplotlib.pyplot as plt
#
#
# # Define the target distribution (you can replace this with your own)
# def target_distribution(x):
#     return np.exp(-0.5 * x ** 2) * np.sin(5 * x)
#
#
# def metropolis_hastings(x_current, x_proposed, beta):
#     acceptance_ratio = min(1, np.exp(beta * (
#             target_distribution(x_current) - target_distribution(
#         x_proposed))))
#     return np.random.rand() < acceptance_ratio
#
#
# def parallel_tempering(num_of_chains, num_of_steps, temperatures):
#     chains = np.random.randn(num_of_chains)
#     sampling = [chains.copy()]
#     for step in range(num_of_steps):
#         for j in range(num_of_chains - 1):
#             beta = temperatures[j] - temperatures[j + 1]
#             if metropolis_hastings(chains[j], chains[j + 1], beta):
#                 chains[j], chains[j + 1] = chains[j + 1], chains[j]
#         sampling.append(chains.copy())
#     return np.array(sampling)
#
# # 1-dimension examples
# num_chains = 5
# num_steps = 1000
# temperatures = np.linspace(1, 5, num_chains)
# samples = parallel_tempering(num_chains, num_steps, temperatures)
#
# plt.figure(figsize=(10, 6))
# for i in range(num_chains):
#     plt.plot(samples[:, i], label=f'Temperature {temperatures[i]:.2f}')
#
# plt.title('Parallel Tempering Chains')
# plt.xlabel('Steps')
# plt.ylabel('Chain Values')
# plt.legend()
# plt.show()
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def target_density(x):
    # Target density function (multimodal)
    term1 = 0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-(x - 6) ** 2 / 2)
    term2 = 1.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-(x + 2) ** 2 / 2)
    return term1 + term2


def metropolis_hastings(x, proposal_std):
    # Metropolis-Hastings proposal
    x_proposed = np.random.normal(x, proposal_std)
    alpha = min(1, target_density(x_proposed) / target_density(x))
    if np.random.rand() < alpha:
        return x_proposed
    else:
        return x


def non_reversible_parallel_tempering(num_chains, num_samples, target_density,
                                      proposal_std, beta_schedule):
    chains = np.zeros((num_chains, num_samples))
    for j in range(num_chains):
        chains[j, 0] = np.random.normal(0, 1)
    for t in range(1, num_samples):
        for i in range(num_chains):
            chains[i, t] = metropolis_hastings(chains[i, t - 1], proposal_std)
        for i in range(num_chains - 1):
            beta = beta_schedule[i]
            acceptance_prob = np.exp((beta_schedule[i + 1] - beta) *
                                     (target_density(
                                         chains[i, t]) - target_density(
                                         chains[i + 1, t])))
            if np.random.rand() < acceptance_prob:
                chains[i, t], chains[i + 1, t] = chains[i + 1, t], chains[i, t]

    return chains



num_chains = 10
num_samples = 10000
proposal_std = 3
beta_schedule = np.linspace(1.0, 10.0, num_chains)  # Linear beta schedule
chains = non_reversible_parallel_tempering(num_chains, num_samples,
                                           target_density, proposal_std,
                                           beta_schedule)
plt.figure(figsize=(10, 6))
for i in range(num_chains):
    plt.plot(chains[i, :], label=f'Chain {i + 1}')

plt.title('Trace Plot of Chains (Non-Reversible Parallel Tempering)')
plt.xlabel('Sample Index')
plt.ylabel('Value')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
x_values = np.linspace(-10, 10, 1000)
target_values = [target_density(x) for x in x_values]
plt.plot(x_values, target_values, label='Target Density', color='red',
         linestyle='dashed')
sns.histplot(chains[0, :], bins=30, stat='density', color='blue',
             label='Sampled Distribution', element='step')

plt.title(
    'Histogram of the First Coordinate with Target Density (Non-Reversible Parallel Tempering)')
plt.xlabel('Value')
plt.ylabel('Probability Density')
plt.legend()
plt.show()
