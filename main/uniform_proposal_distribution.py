import numpy as np
import matplotlib.pyplot as plt


def target_distribution(x, f):
    """Define the target distribution (i.i.d.) as the product form"""
    if isinstance(x, int) or isinstance(x, float):
        return f(x)
    else:
        multi_dim_target = 1
        for item in x:
            multi_dim_target *= f(item)  # iid target distribution
        return np.array(multi_dim_target)


def proposal_distribution(dimension, x, proposed_cov, function):
    """Define the proposal distribution of the Metropolis Algorithm"""
    if function == 'multivariate' and (
            isinstance(x, int) or isinstance(x, float)):
        step = np.random.uniform(x - 0.5 * proposed_cov, x + 0.5 * proposed_cov)
        return step

    elif function == 'multivariate' and not (
            isinstance(x, int) or isinstance(x, float)):
        step = np.random.uniform(x - 0.5 * proposed_cov, x + 0.5 * proposed_cov, size=(dimension))
    else:
        step = []
        for _ in range(dimension):
            step.append(function())
    return np.array(step)


def metropolis_algorithm(dimension, target_distrn, proposed_distribution,
                         iteration, proposed_cov, iid=True):
    if dimension != 1:
        x_initial = np.zeros(dimension) # the initial starting points can be modified
    else:
        x_initial = 0
    sampling = [np.array(x_initial)]
    x_0 = x_initial
    esjd = 0
    acc_rate = 0
    for _ in range(iteration):
        propos = proposal_distribution(dimension, x_0, proposed_cov,
                                       proposed_distribution)
        y = propos
        prev = target_distribution(x_0, target_distrn)
        if prev.item() != 0:
            acceptance_ratio = target_distribution(y, target_distrn) / prev
        else:
            acceptance_ratio = 1
        u = np.random.uniform(0, 1)
        if u < acceptance_ratio:
            acc_rate += 1
            esjd += np.sum(np.square(np.array(x_0) - np.array(y)))
            x_0 = y
        sampling.append(np.array(x_0))
    return np.array(sampling), esjd / iteration, acc_rate / iteration


def trace_plot(sampling, dimension):
    if dimension == 1:
        plt.plot(sampling)
        plt.xlabel("Iteration")
        plt.title("Trace Plot")
        plt.show()
    else:
        for i in range(dimension):
            lst = []
            for j in range(0, len(sampling)):
                lst.append(sampling[j][i])
            plt.plot(lst)
            plt.xlabel("Iteration")
            plt.title(f"Trace Plot for {i + 1}th coordinates")
            plt.show()


def target_distrn_1_dim(x):
    # for each single element, and assume iid
    # normal distribution for each element
    target = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (x ** 2))
    return target


def hist(sampling, dimension):
    for i in range(dimension):
        x = np.linspace(np.min(sampling[:, i]), np.max(sampling[:, i]), 10000)
        plt.plot(x, target_distrn_1_dim(x), color='red', linewidth=2)
        plt.hist(sampling[:, i], bins=50, density=True, alpha=0.7)
        plt.title(f'Metropolis Algorithm with proposal variance 1.2')
        plt.xlabel('Value')
        plt.ylabel('Probability Density')
        plt.legend()
        plt.show()


#Running experiments
rate = [0.01, 0.05, 0.1, 0.23, 0.4, 0.5, 0.7, 1, 1.2, 1.5, 2, 4, 4.6, 5, 6, 6.5]
# rate=[1.2]

lst_jump = []
acc = []
number_iter = 100000
dimension = 20

# Generating the histogram and trace plot for each coordinate
# for i in rate:
#     samples, esjd, acc_rate = metropolis_algorithm(dimension, target_distrn_1_dim,
#                                          'multivariate', number_iter, i)
#     plt.ylabel(f"x with proposed variance {i}")
#     trace_plot(samples[2500:], dimension)
#     hist(samples, dimension)
#     lst_jump.append(esjd)
#     acc.append(acc_rate)

for i in rate:
    plt.figure(figsize=(12, 8))
    samples, esjd, acc_rate = metropolis_algorithm(dimension, target_distrn_1_dim,
                                                   'multivariate', number_iter, i)
    print(f"Acceptance rate for rate {i}: {acc_rate}")
    for dim in range(dimension):
        plt.subplot(dimension, 1, dim + 1)
        plt.plot(samples[:, dim], label=f'Dimension {dim + 1}')
        plt.title(f'Trace Plot for Rate {i}, Dimension {dim + 1}')
        plt.xlabel('Iteration')
        plt.ylabel('Value')
        plt.legend()
    plt.tight_layout()
    plt.show()
    lst_jump.append(esjd)
    acc.append(acc_rate)

# Relationship between ESJD and proposal variance
plt.scatter(rate, lst_jump, color='blue')
plt.plot(rate, lst_jump, color='red', linestyle='-', marker='o', label='Connected Line')
plt.xlabel('proposed variance')
plt.ylabel('ESJD')
plt.show()

# Relationship between ESJD and acceptance rate
plt.scatter(acc, lst_jump, color='skyblue')
plt.plot(acc, lst_jump, color='skyblue', linestyle='-', marker='o', label='Connected Line')
plt.xlabel('acceptance rate')
plt.ylabel('ESJD')
plt.show()
