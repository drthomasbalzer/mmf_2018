################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def comparison_poi_binom(lam, upper_bound):

    check_sum = 0
    n = upper_bound + 1
    x_ax = [0.] * n
    n_plots = 2  # plotting both poisson and binomial distribution
    y_ax = [0.] * n_plots
    for k in range(n_plots):
        y_ax[k] = [0.] * n

    for k in range(0, n):
        x_ax[k] = k
        y_ax[0][k] = dist.poisson_pdf(lam, k)  # poisson distribution
        y_ax[1][k] = dist.binomial_pdf(lam / n, k, n)  # binomial distribution
        check_sum = check_sum + y_ax[0][k]

    mp = pu.PlotUtilities("Poisson Vs Binomial distribution for lambda={0}".format(lam), "# Successes", "Probability")
    mp.multiPlot(x_ax, y_ax, '*')

    print (check_sum)

def poisson_plot(lam, upper_bound):

    n = upper_bound + 1
    x_ax = [0.] * n
    y_ax = [[0.] * n]
    for k in range(0, n):
        x_ax[k] = k
        y_ax[0][k] = dist.poisson_pdf(lam, k)

    mp = pu.PlotUtilities("Poisson Distribution for lambda={0}".format(lam), "# Successes", "Probability")
    mp.multiPlot(x_ax, y_ax, 'o')

def binomial_plot(p, n):

    check_sum = 0
    x_ax = [0.] * n
    y_ax = [[0.] * n]

    for k in range(0, n):
        p_k = dist.binomial_pdf(p, k, n)
        x_ax[k] = k
        y_ax[0][k] = dist.binomial_pdf(p, k, n)
        check_sum = y_ax[0][k]

    mp = pu.PlotUtilities("Binomial Distribution for p={0}".format(p), "# Successes", "Probability")
    mp.multiPlot(x_ax, y_ax, 'o')

    print (check_sum)

#####
## Create distribution via Quantile Transform -- $B(n,p)$- vs $Poi(\lambda)$-distribution
#####

def bin_vs_poisson_histogram(_lambda, n, sz):

    lower_bound = 0.
    upper_bound = 1.

    p = _lambda / n
    total_sample_size = n * sz
    print total_sample_size
    uni_sample = np.random.uniform(lower_bound, upper_bound, total_sample_size)

    #######
    ### transform the uniform sample
    #######
    sample = [0.] * total_sample_size
    for j in range(total_sample_size):
        sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

    outcome_bnp = [0.] * sz
    index = 0
    for k in range(sz):
        for l in range(n):
            outcome_bnp[k] = outcome_bnp[k] + sample[index]
            index = index + 1

    # the histogram of the data

    num_bins = 30
    hp = pu.PlotUtilities("Histogram of B(n,p) sample with p={0} and n={1}".format(p, n), 'Outcome', 'Rel. Occurrence')
    hp.plotHistogram(outcome_bnp, num_bins)

if __name__ == '__main__':

#    binomial_plot(0.2, 50)
#    poisson_plot(0.5, 10)
#   comparison_poi_binom(0.5,10)
    _lambda = 0.5
    n = 20
    sz = 10000
    bin_vs_poisson_histogram(_lambda, n, sz)


