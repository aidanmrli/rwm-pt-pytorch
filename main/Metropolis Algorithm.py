import numpy as np
import matplotlib.pyplot as plt
<<<<<<<< HEAD:main/Metropolis Algorithm.py
from scipy.stats import multivariate_normal
========
>>>>>>>> 2e5fd586d2457e9e2481faa3feaa32efc3cf8ff2:algorithms/Metropolis Algorithm.py
from scipy.stats import beta as beta_dist


def target_distribution(x, f):
    if isinstance(x, int) or isinstance(x, float):
        return f(x)
    else:
        multi_dim_target = 1
        for item in x:
            multi_dim_target *= f(item)  # iid
        return np.array(multi_dim_target)

    # mean1 = np.concatenate(([dimension], np.zeros(dimension-1)))
    # mean2 = np.concatenate((np.zeros(dimension-1), [dimension]))
    # covariance_matrix = np.eye(dimension)
    # component1 = 0.5 * multivariate_normal(mean1, covariance_matrix).pdf(x)
    # component2 = 0.5 * multivariate_normal(mean2, covariance_matrix).pdf(x)
    # mixture_pdf = component1 + component2
    # return np.array(mixture_pdf)


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
    accept = 0
    if dimension != 1:
        # x_initial = np.random.uniform(-10, 10, size=dimension)
        x_initial = np.zeros(dimension)
    else:
        x_initial = np.random.uniform(-10, 10)
        # x_initial = 0
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
            accept += 1
            x_0 = y
        sampling.append(x_0)
    acceptance_rate = accept / iteration
    return sampling, esjd, acceptance_rate


def trace_plot(sampling, dimension):
    if dimension == 1:
        plt.plot(sampling)
        plt.xlabel("Iteration")
        plt.title("Trace Plot")
        plt.show()
    else:
<<<<<<<< HEAD:main/Metropolis Algorithm.py
        # for i in range(dimension):
        lst = []
        for j in range(0, len(sampling)):
            lst.append(sampling[j][0])
        plt.plot(lst)
        plt.xlabel("Iteration")
        plt.title(f"Trace Plot for 1st coordinates")
        plt.show()
========
        for i in range(dimension):
            lst = []
            for j in range(0, len(sampling)):
                lst.append(sampling[j][i])
            plt.plot(lst)
            plt.xlabel("Iteration")
            plt.title(f"Trace Plot for {i + 1}th coordinates")
            plt.show()
>>>>>>>> 2e5fd586d2457e9e2481faa3feaa32efc3cf8ff2:algorithms/Metropolis Algorithm.py


def target_distrn_1_dim(x):
    # for each single element, and assume iid

<<<<<<<< HEAD:main/Metropolis Algorithm.py
    # # normal distribution for each element
    target = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (x ** 2))

    # multi gaussian
    # target = 0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 6) ** 2) + \
    #          1.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 2) ** 2)

========
    # normal distribution for each element
    target = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (x ** 2))

    # multi gaussian
    # target = 0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 6)**2) + \
    #          0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 2)**2) + \
    #          3*(0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 7)**2))

    # beta
    # target = beta_dist.pdf(x, 2, 5)
>>>>>>>> 2e5fd586d2457e9e2481faa3feaa32efc3cf8ff2:algorithms/Metropolis Algorithm.py
    return target


# rate = [0.001,0.005,0.01,0.05,0.1,0.2, 0.4, 0.5,1, 1.5,2]
rate = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.5, 1, 1.5, 2, 2.5, 3, 4]
dimension = 100
# rate = [0.01]
# rate = [0.2]
lst_jump = []
<<<<<<<< HEAD:main/Metropolis Algorithm.py
number_iter = 20000
acc = []
for i in rate:
    # 1 dimension, dimension = 1
    samples, esjd, acceptance_rate = metropolis_algorithm(dimension,
                                                          target_distrn_1_dim,
                                                          'multivariate',
                                                          number_iter, i)
    # plt.ylabel(f"x with proposed variance {i}")
    # trace_plot(samples, dimension)
    lst_jump.append(esjd / number_iter)
    acc.append(acceptance_rate)
    # plt.figure(figsize=(10, 6))
    # x = np.linspace(-10, 10, 5000)
    # histogram_lst = []
    # lst = []
    # if dimension != 1:
    #     for j in range(0, len(samples)):
    #         lst.append(samples[j][0])
    # else:
    #     lst = samples
    # for num in x:
    #     histogram_lst.append(target_distribution(num, target_distrn_1_dim))
    # plt.plot(x, histogram_lst, label='Target Distribution', linewidth=2)
    # plt.hist(lst, bins=50, density=True, alpha=0.7,
    #          label='Generated Samples')
    # plt.title(f'Metropolis Algorithm x with proposed variance {i}')
    # plt.xlabel('Samples')
    # plt.ylabel('Density')
    # plt.show()

plt.plot(acc, lst_jump, 'd--')
plt.xlabel('acceptance rate')
plt.ylabel('ESJD')
plt.legend()
plt.show()

========
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
>>>>>>>> 2e5fd586d2457e9e2481faa3feaa32efc3cf8ff2:algorithms/Metropolis Algorithm.py
# plt.plot(rate, lst_jump)
# plt.xlabel('proposed variance')
# plt.ylabel('ESJD')
# plt.show()
<<<<<<<< HEAD:main/Metropolis Algorithm.py
#
# plt.plot(rate, acc)
# plt.xlabel('rate')
# plt.ylabel('acc')
# plt.show()
========

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
>>>>>>>> 2e5fd586d2457e9e2481faa3feaa32efc3cf8ff2:algorithms/Metropolis Algorithm.py
