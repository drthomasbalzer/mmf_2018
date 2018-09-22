################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist
import plot_utilities as pu

###########
##### Demo of Quantile Transformation
###########

def uniform_histogram(sz):

    lower_bound = 0.
    upper_bound = 1.

    sample = np.random.uniform(lower_bound, upper_bound, sz)

    num_bins = 50

    hp = pu.PlotUtilities("Histogram of Uniform Sample of Size={0}".format(sz), 'Outcome', 'Rel. Occurrence')
    hp.plotHistogram(sample, num_bins)

#####
## Create distribution via Quantile Transform -- $B(1,p)$-distribution
#####

def binomial_histogram(p, sz):

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    #######
    ### transform the uniform sample
    #######
    sample = [0.] * sz
    for j in range(sz):
        sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

    num_bins = 100

    hp = pu.PlotUtilities("Histogram of Binomial Sample with Success Probability={0}".format(p), 'Outcome', 'Rel. Occurrence')
    hp.plotHistogram(sample, num_bins)

#####
## Create distribution via Quantile Transform -- $Exp(\lambda)$ distribution
#####


def exponential_histogram(_lambda, sz):

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    #######
    ### transform the uniform sample
    #######
    sample = [0.] * sz
    for j in range(sz):
        sample[j] = dist.exponential_inverse_cdf(_lambda,uni_sample[j])

    num_bins = 50

    # the histogram of the data
    n, bins, _hist = plt.hist(sample, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Exponential Sample with Parameter={0}".format(_lambda))

    y = [0.] * (num_bins+1)
    ###### overlay the actual pdf
    for i in range(0,num_bins+1):
         y[i] = dist.exponential_pdf(_lambda, bins[i])

    plt.plot(bins, y, 'r--')
    # # Tweak spacing to prevent clipping of ylabel
    plt.subplots_adjust(left=0.15)
    plt.show()


if __name__ == '__main__':

    calc_type = 2

    size = 10000

    if (calc_type == 0):  ### uniform sample
        uniform_histogram(size)
    else:
        if (calc_type == 1): ### generate a binomial distribution
            p = 0.40
            binomial_histogram(p, size)
        else: ### generate an exponential distribution
            _lambda = 1.
            exponential_histogram(_lambda, size)

