import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from tqdm import tqdm


def target_distribution(x):
    """Define the target distribution as the product of the density functions
    f_i of each coordinate. Here, we define the dimension = 50, and the mean of
    each f_i is 1/5i, and the variance is 1"""
    product = 1
    for i in range(1, 51):
        mu = 1 / (5 * i)
        pdf_i = norm.pdf(x[i - 1], loc=mu, scale=1)
        product *= pdf_i
    return product


def proposal_distribution(dimension, proposed_cov):
    """Define proposal distribution Q(0, proposed_cov I_d)"""
    step = np.random.multivariate_normal(np.zeros(dimension),
                                         proposed_cov * np.eye(dimension))
    return step


def trace_plot(sampling, dimension, coordinate=None):
    """Generate Trace Plot of a specific coordinate"""
    if dimension == 1:
        plt.plot(sampling)
        plt.xlabel("Iteration")
        plt.title("Trace Plot")
        plt.show()
    else:
        lst = []
        if coordinate is None:
            print("Please provide a coordinate for trace plot.")
            return
        for j in range(len(sampling)):
            lst.append(sampling[j][coordinate])
        plt.plot(lst)
        plt.xlabel("Iteration")
        plt.title(f"Trace Plot for {coordinate}th coordinate")
        plt.show()


def target_distrn_1_dim(x, mu):
    # for each single element, and assume iid
    target = 1 / np.sqrt(2 * np.pi) * np.exp(-0.5 * (x - mu) ** 2)
    return target


def metropolis_algorithm(dimension, target_distrn, iteration, proposed_cov):
    """Running metropolis algorithm"""
    accept = 0
    x_initial = np.zeros(dimension)  # Starting point
    sampling = [x_initial]
    x_0 = x_initial
    esjd = 0
    for _ in tqdm(range(iteration)):
        propos = proposal_distribution(dimension, proposed_cov)
        y = x_0 + propos
        prev = target_distrn(x_0)
        acceptance_ratio = target_distrn(y) / prev
        u = np.random.uniform(0, 1)
        if u < acceptance_ratio:
            esjd += np.sum(np.square(x_0 - y))
            accept += 1
            x_0 = y
        sampling.append(x_0)
    acceptance_rate = accept / iteration
    return sampling, esjd, acceptance_rate


"""These are the experiments I've done for this case"""

rate = np.linspace(0.05, 0.5, 15)
additional_values = np.linspace(0.6, 1, 15)
rate = np.append(rate, additional_values)
# rate = [0.114]
dimension = 50
number_iter = 200000
lst_jump = []
acc = []
coordinate_of_interest = 40  # Change this to the desired coordinate
specific_mu = 1 / (5 * coordinate_of_interest)
for i in rate:
    samples, esjd, acceptance_rate = metropolis_algorithm(dimension,
                                                          target_distribution,
                                                          number_iter, i)
    true_esjd = esjd / number_iter
    lst_jump.append(true_esjd)
    print((i, acceptance_rate, true_esjd))
    acc.append(acceptance_rate)
    trace_plot(samples, dimension,
               coordinate=coordinate_of_interest)  # Trace plot for the specified coordinate
    x = np.linspace(-10, 10, 5000)
    histogram_lst = []
    lst = []
    for j in range(len(samples)):
        lst.append(samples[j][coordinate_of_interest])
    for index in range(len(x)):
        histogram_lst.append(target_distrn_1_dim(x[index], specific_mu))
    plt.plot(x, histogram_lst, label='Target Distribution', linewidth=2)
    plt.hist(lst, bins=50, density=True, alpha=0.7, label='Generated Samples')
    plt.title(f'Metropolis Algorithm for x with proposed variance {i}')
    plt.xlabel('Samples')
    plt.ylabel('Density')
    plt.legend()
    plt.show()

# ESJD vs. acceptance rate
plt.plot(acc, lst_jump, 'd--')
plt.xlabel('Acceptance rate')
plt.ylabel('ESJD')
plt.show()

# ESJD vs proposed variance
# plt.plot(rate, lst_jump, 'd--')
# plt.xlabel('Proposed variance')
# plt.ylabel('ESJD')
# plt.show()

