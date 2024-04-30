import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
from tqdm import tqdm



def generate_covariance_matrix(dim):
    """Define the covariance matrix of target distribution, dependent case"""
    np.random.seed(40)
    M = np.random.normal(0, 2, size=(dim, dim))
    Sigma_star = np.dot(M, M.T)
    return Sigma_star


def metropolis_algorithm(target_density, initial_sample, iterations,
                         sigma_squared):
    """Running metropolis algorithm with different proposal distribution Q(x)
    Case 1. the covariance matrix of Q is sigma^2 I_d
    Case 2. the covariance matrix of Q depends on Sigma_star, which is the covariance matrix of target distribution
    Case 3. Using adaptive RWM to define the covariance matrix of Q """
    samples = [initial_sample]
    current_sample = initial_sample
    esjd = 0
    accept = 0
    cov_matrix = None
    for times in tqdm(range(iterations)):
        # Propose a new sample with Case 1
        # proposal = current_sample + np.random.multivariate_normal(np.zeros(30), sigma_squared * np.eye(30))

        # Propose a new sample with Case 2
        # proposal = current_sample + np.random.multivariate_normal(np.zeros(dim), sigma_squared*Sigma_star)

        # Propose a new sample with Case 3
        if accept <= 10:
            proposal = current_sample + np.random.normal(0, np.sqrt(sigma_squared), size=len(current_sample))
        else:
            proposal = current_sample + np.random.multivariate_normal(np.zeros(len(current_sample)), sigma_squared * cov_matrix)
        acceptance_ratio = min(1, target_density(proposal) / target_density(
            current_sample))
        if np.random.rand() < acceptance_ratio:
            esjd += np.sum(
                np.square(np.array(current_sample) - np.array(proposal)))
            current_sample = proposal
            accept += 1
            if accept <= 10:
                cov_matrix =  np.eye(len(initial_sample))
            else:
                # update covariance matrix recursively
                cov_matrix = (accept / (accept + 1)) * cov_matrix + (1 / (accept + 1)) * np.outer(current_sample - np.mean(samples, axis=0), current_sample - np.mean(samples, axis=0))
        # if cov_matrix is None:
        #     cov_matrix = np.eye(len(initial_sample))
        samples.append(current_sample)
    acceptance_rate = accept / iterations
    return np.array(samples), esjd, acceptance_rate


def target_distrn_1_dim(x, var_1_dim):
    """distribution of each coordinate in the target distribution"""
    target = 1 / (np.sqrt(var_1_dim)*np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x/np.sqrt(var_1_dim)) ** 2))
    return target


# Run the code
dim = 30
lst_jump = []
acc = []
Sigma_star = generate_covariance_matrix(dim)
target_density = stats.multivariate_normal(mean=np.zeros(dim),
                                           cov=Sigma_star).pdf

initial_sample = np.zeros(dim)
iterations = 200000

inirate =np.linspace(0.001, 0.01, 10)
addrate = np.linspace(0.01, 0.9, 50)
rate = np.append(inirate, addrate)
print(rate)

# rate = [0.04]
want_dim = 10
var_hist = Sigma_star[want_dim-1, want_dim-1]
for sigma_squared in rate:
    samples, esjd, acceptance_rate = metropolis_algorithm(target_density,
                                                          initial_sample,
                                                          iterations,
                                                          sigma_squared)
    true_esjd = esjd/iterations
    print((sigma_squared,acceptance_rate, true_esjd))
    lst_jump.append(true_esjd)
    acc.append(acceptance_rate)

    # Plot the Trace Plot of the want_dim coordinate for the corresponding sigma_squared
    plt.plot(samples[:, want_dim-1])
    plt.title(f'Dimension {want_dim} with proposed variance {sigma_squared}')
    plt.xlabel('Iterations')
    plt.ylabel('Sample Value')
    plt.show()

    # Plot the histogram of the want_dim coordinate for the corresponding sigma_squared
    x = np.linspace(-50, 50, 50000)
    histogram_lst = []
    lst = []
    if dim != 1:
        for j in range(0, len(samples)):
            lst.append(samples[j][want_dim-1])
    else:
        lst = samples
    for num in x:
        histogram_lst.append(target_distrn_1_dim(num, var_hist))
    plt.plot(x, histogram_lst, label='Target Distribution', linewidth=2)
    plt.hist(lst, bins=50, density=True, alpha=0.7,
             label='Generated Samples')
    plt.title(f'Metropolis Algorithm x with proposed variance {sigma_squared}')
    plt.xlabel('Samples')
    plt.ylabel('Density')
    plt.show()


# ESJD vs. acceptance rate
plt.plot(acc, lst_jump,'d--')
plt.xlabel('acceptance rate')
plt.ylabel('ESJD')
plt.show()

# ESJD vs. proposed variance
plt.plot(rate, lst_jump, 'd--')
plt.xlabel('proposed variance')
plt.ylabel('ESJD')
plt.show()
