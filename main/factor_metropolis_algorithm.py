import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from scipy.stats import beta as beta_dist


def target_distribution(x, f, factor):
    if isinstance(x, int) or isinstance(x, float):
        return f(x, factor[0])
    else:
        multi_dim_target = 1
        for k in range(0, len(x)):
            multi_dim_target *= f(x[k], factor[k]) * factor[k]  # iid
        return np.array(multi_dim_target)


def proposal_distribution(dimension, x, proposed_cov, function, factor, xmin=-5,
                          xmax=5):
    # symmetric
    # default: Q = N(0, \Sigma), where \Sigma = proposed_cov * identity matrix
    if function == 'multivariate' and (
            isinstance(x, int) or isinstance(x, float)):
        step = np.random.normal(0, proposed_cov)
        return step

    elif function == 'multivariate' and not (
             isinstance(x, int) or isinstance(x, float)):
        # step = np.random.multivariate_normal(np.zeros(dimension),
        #                                      proposed_cov * np.eye(dimension))
        step = np.random.multivariate_normal(np.zeros(dimension), proposed_cov * np.diag(factor))
    else:
        step = []
        for _ in range(dimension):
            step.append(function())
            # step.append(function(np.random.uniform(xmin, xmax)))
    return np.array(step)


def metropolis_algorithm(dimension, target_distrn, proposed_distribution,
                         iteration, proposed_cov, factor, iid=True):
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
                                       proposed_distribution, factor)
        y = x_0 + propos
        prev = target_distribution(x_0, target_distrn, factor)
        if prev.item() != 0:
            acceptance_ratio = target_distribution(y, target_distrn, factor) / prev
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
        # for i in range(dimension):
        lst = []
        for j in range(0, len(sampling)):
            lst.append(sampling[j][16])
        plt.plot(lst)
        plt.xlabel("Iteration")
        plt.title(f"Trace Plot for 1st coordinates")
        plt.show()


def target_distrn_1_dim(x, factor=1):
    # for each single element, and assume iid
    target = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * ((factor * x) ** 2))
    # multi gaussian
    # target = 0.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x - 6) ** 2) + \
    #          1.5 * (1 / np.sqrt(2 * np.pi)) * np.exp(-0.5 * (x + 2) ** 2)

    return target


rate = [0.005,0.01,0.02, 0.03, 0.05, 0.07, 0.1,0.15, 0.2, 0.25, 0.3, 0.4, 0.5,0.7, 1, 1.5,2, 3]
# rate = [0.01, 0.05, 0.1, 0.2, 0.3, 0.45, 0.5, 1, 1.5, 2, 2.5, 3, 4]
dimension = 30
# rate = [0.15]
# rate = [0.2]
lst_jump = []
number_iter = 20000
acc = []
scaling_factors = np.random.uniform(0.2, 1.8, dimension)
# scaling_factors = np.random.exponential(scale=1, size=dimension)
print(scaling_factors[0])
for i in rate:
    # 1 dimension, dimension = 1
    samples, esjd, acceptance_rate = metropolis_algorithm(dimension,
                                                          target_distrn_1_dim,
                                                          'multivariate',
                                                          number_iter, i, scaling_factors)
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
    #         lst.append(samples[j][16])
    # else:
    #     lst = samples
    # for index in range(0, len(x)):
    #     histogram_lst.append(target_distribution(x[index], target_distrn_1_dim, [scaling_factors[16]]))
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
plt.show()

plt.plot(rate, lst_jump)
plt.xlabel('proposed variance')
plt.ylabel('ESJD')
plt.show()

# plt.plot(rate, acc)
# plt.xlabel('rate')
# plt.ylabel('acc')
# plt.show()
