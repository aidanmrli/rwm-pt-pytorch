import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import multivariate_normal
from scipy.stats import beta as beta_dist


def target_distribution(x, f, factor):
    """product of density function of each coordinate, with scaling factors"""
    if isinstance(x, int) or isinstance(x, float):
        return f(x, factor[0]) * factor[0]
    else:
        multi_dim_target = 1
        for k in range(0, len(x)):
            multi_dim_target *= f(x[k], factor[k]) * factor[k]  # iid
        return np.array(multi_dim_target)


def proposal_distribution(dimension, x, proposed_cov, function, factor, xmin=-5,
                          xmax=5):
    """Conduct different proposal distribution
    Case 1: without scaling factors: Q(x) = N(0, sigma^2 I_d)
    Case 2: scale each coordinate in the same extent as th target distribution
    Q(x) = N(0, sigma^2 diag(C1, ... C_d))"""
    if function == 'multivariate' and (
            isinstance(x, int) or isinstance(x, float)):
        step = np.random.normal(0, proposed_cov)
        return step

    elif function == 'multivariate' and not (
             isinstance(x, int) or isinstance(x, float)):
        # Case 1
        # step = np.random.multivariate_normal(np.zeros(dimension), proposed_cov * np.eye(dimension))

        # Case 2
        step = np.random.multivariate_normal(np.zeros(dimension), proposed_cov * np.diag(factor))
    else:
        step = []
        for _ in range(dimension):
            step.append(function())
            # step.append(function(np.random.uniform(xmin, xmax)))
    return np.array(step)


def metropolis_algorithm(dimension, target_distrn, proposed_distribution,
                         iteration, proposed_cov, factor, iid=True):
    """Conduct metropolis algorithm, with factor as a parameter"""
    accept = 0
    if dimension != 1:
        # x_initial = np.random.uniform(-10, 10, size=dimension)
        x_initial = np.zeros(dimension)
    else:
        # x_initial = np.random.uniform(-10, 10)
        x_initial = 0
    sampling = [x_initial]
    x_0 = x_initial
    esjd = 0
    for _ in range(iteration):
        propos = proposal_distribution(dimension, x_0, proposed_cov,
                                       proposed_distribution, factor)
        y = x_0 + propos
        prev = target_distribution(x_0, target_distrn, factor)
        if prev.item() != 0:
            acceptance_ratio = target_distribution(y, target_distrn, factor) / prev
        else:
            acceptance_ratio = 1
        u = np.random.uniform(0, 1)
        if u <= acceptance_ratio:
            # esjd += np.sum(np.square(np.array(x_0) - np.array(y)))

            esjd += np.sum(np.square(np.array(x_0[0]) - np.array(y[0])))

            # esjd += np.sum(np.square(np.array(x_0[2]) - np.array(y[2])))

            accept += 1
            x_0 = y
        sampling.append(x_0)
    acceptance_rate = accept / iteration
    return sampling, esjd, acceptance_rate


def trace_plot(sampling, dimension, coord):
    if dimension == 1:
        plt.plot(sampling)
        plt.xlabel("Iteration")
        plt.title("Trace Plot")
        plt.show()
    else:
        # for i in range(dimension):
        lst = []
        for j in range(0, len(sampling)):
            lst.append(sampling[j][coord-1])
        plt.plot(lst)
        plt.xlabel("Iteration")
        plt.title(f"Trace Plot for {coord} coordinates")
        plt.show()


def target_distrn_1_dim(x, factor):
    # for each single element, and assume iid
    target = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((factor * x) ** 2))
    return target


"""Below are the experiments I generate"""

# rate = np.linspace(0.01, 0.4, 28)
# additional_values = np.array([0.5, 0.6, 0.7, 0.8, 1, 1.5, 2, 2.5, 3, 3.5, 4])
# rate = np.append(rate, additional_values)
rate = [0.005, 0.01, 0.05, 0.1, 0.15, 0.16, 0.17, 0.18, 0.2, 0.25, 0.3, 0.45, 0.5, 1, 1.5, 2, 2.5, 3, 4]
# rate = [0.165]

# rate = [0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.5, 1, 1.5, 2, 2.5, 3, 4]
dimension = 30
# rate = [0.15]
# rate = [0.204]
lst_jump = []
number_iter = 100000
acc = []
coord = 13
# scaling_factors = np.ones(dimension)  # 过关
# scaling_factors = np.random.uniform(low=0.2, high=1.8, size=dimension)

# Known scaing factors
scaling_factors =[1,    1.63536268, 0.20606286, 0.75356805, 1.0141207 ,
                  1.36688048, 1.74929357, 0.48,        0.44485666, 0.90017706,
                  0.8739331 , 1.05461719, 1.68314939, 1.07756459, 1.15740694,
                  0.86954708, 1.29618255, 0.72106731, 1.62659929, 1.07590837,
                  0.56254577, 1.1778348 , 0.68447311, 1.2061486,  0.8959104,
                  0.29360274, 0.63138624, 1.71038852, 0.62893131, 1.39748293]
# scaling_factors = np.random.exponential(scale=1, size=dimension)
# scaling_factors[0] = 1
print(scaling_factors)
for i in rate:
    samples, esjd, acceptance_rate = metropolis_algorithm(dimension,
                                                          target_distrn_1_dim,
                                                          'multivariate',
                                                          number_iter, i,
                                                          scaling_factors)
    true_esjd = esjd / number_iter
    lst_jump.append(true_esjd)
    acc.append(acceptance_rate)
    print(i, true_esjd, acceptance_rate)
    # plt.ylabel(f"x with proposed variance {i}")
    # trace_plot(samples, dimension, coord)
    # x = np.linspace(-6, 6, 10000)
    # histogram_lst = []
    # lst = []
    # if dimension != 1:
    #     for j in range(0, len(samples)):
    #         lst.append(samples[j][coord-1])
    # else:
    #     lst = samples
    # for index in range(0, len(x)):
    #     histogram_lst.append(target_distribution(x[index], target_distrn_1_dim, [scaling_factors[coord-1]]))
    # plt.plot(x, histogram_lst, label='Target Distribution', linewidth=2)
    # plt.hist(lst, bins=50, density=True, alpha=0.7,
    #          label='Generated Samples')
    # plt.title(f'Metropolis Algorithm x with proposed variance {i}')
    # plt.xlabel('Samples')
    # plt.ylabel('Density')
    # plt.show()

# ESJD vs. acceptance rate
plt.plot(acc, lst_jump, 'd--')
plt.xlabel('acceptance rate')
plt.ylabel('ESJD')
plt.show()

# ESJD vs. proposed variance
plt.plot(rate, lst_jump, 'd--')
plt.xlabel('proposed variance')
plt.ylabel('ESJD')
plt.show()


