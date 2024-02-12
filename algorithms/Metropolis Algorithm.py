import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist


def target_distribution(x, f):
    if isinstance(x, int) or isinstance(x, float):
        return f(x)
    else:
        multi_dim_target = 1
        for item in x:
            multi_dim_target *= f(item)  # iid
        return np.array(multi_dim_target)


def proposal_distribution(dimension, x, proposed_cov, function, xmin=-5,
                          xmax=5):
    # symmetric
    # default: Q = N(0, \Sigma), where \Sigma = proposed_cov * identity matrix
    if function == 'multivariate' and (
            isinstance(x, int) or isinstance(x, float)):
        step = np.random.normal(0, proposed_cov)
        return step

    elif function == 'multivariate' and not (
            isinstance(x, int) or isinstance(x, float)):
        step = np.random.multivariate_normal(np.zeros(dimension),
                                             proposed_cov * np.eye(dimension))
    else:
        step = []
        for _ in range(dimension):
            step.append(function())
            # step.append(function(np.random.uniform(xmin, xmax)))
    return np.array(step)


def initial_state(xmin, xmax, dimension):
    x_lst = []
    for _ in range(dimension):
        x_lst.append(np.random.uniform(xmin, xmax))
    return x_lst


def metropolis_algorithm(dimension, target_distrn, proposed_distribution,
                         iteration, proposed_cov, iid=True):
    # set by default, you could change initial point if you want
    if dimension != 1:
        x_initial = np.zeros(dimension)
    else:
        x_initial = 0
    sampling = [x_initial]
    x_0 = x_initial
    esjd = 0
    for _ in range(iteration):
        propos = proposal_distribution(dimension, x_0, proposed_cov,
                                       proposed_distribution)
        y = x_0 + propos
        prev = target_distribution(x_0, target_distrn)
        if prev.item() != 0:
            acceptance_ratio = target_distribution(y, target_distrn) / prev
        else:
            acceptance_ratio = 1
        u = np.random.uniform(0, 1)
        if u < acceptance_ratio:
            esjd += np.sum(np.square(np.array(x_0) - np.array(y)))
            x_0 = y
        sampling.append(x_0)
    return sampling, esjd


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

    # multi gaussian
    # target = 0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 6)**2) + \
    #          0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 2)**2) + \
    #          3*(0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 7)**2))

    # beta
    # target = beta_dist.pdf(x, 2, 5)
    return target


# def other_proposal_distribution():
#     proposal_value = np.random.uniform(-1, 1)
#     return proposal_value


# Univariate Proposal Distribution
# def assigned_proposal(degrees_of_freedom=3):
#     step = np.random.standard_t(degrees_of_freedom)
#     return step


rate = [0.01, 0.05, 0.1, 0.23, 0.4, 0.5, 0.7, 1, 2, 4, 4.6, 5]

lst_jump = []
dimension = 1
number_iter = 4000

# for i in rate:
#     # 1 dimension, dimension = 1
#     samples, esjd = metropolis_algorithm(dimension, target_distrn_1_dim,
#                                          'multivariate', number_iter, i)
#     plt.ylabel(f"x with proposed variance {i}")
#     trace_plot(samples, dimension)
#     lst_jump.append(esjd)
#     plt.figure(figsize=(10, 6))
#     x = np.linspace(-5, 5, 5000)
#     histogram_lst = []
#     for num in x:
#         histogram_lst.append(target_distribution(num, target_distrn_1_dim))
#     plt.plot(x, histogram_lst, label='Target Distribution', linewidth=2)
#     plt.hist(samples, bins=50, density=True, alpha=0.7,
#              label='Generated Samples')
#     plt.title(f'Metropolis Algorithm x with proposed variance {i}')
#     plt.xlabel('Samples')
#     plt.ylabel('Density')
#     plt.show()
#
# plt.plot(rate, lst_jump)
# plt.xlabel('proposed variance')
# plt.ylabel('ESJD')
# plt.show()

dimension = 3
for i in rate:
    # high dimension, dimension = 10
    samples, esjd = metropolis_algorithm(dimension, target_distrn_1_dim,
                                         'multivariate', number_iter, i)
    plt.ylabel(f"x with proposed variance {i}")
    trace_plot(samples, dimension)
    lst_jump.append(esjd)

plt.plot(rate, lst_jump)
plt.xlabel('proposed variance')
plt.ylabel('ESJD')
plt.show()
