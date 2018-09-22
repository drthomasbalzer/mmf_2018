################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu
import matplotlib.pyplot as plt

###########
##
## Demo of the Law of Large Numbers
##
###########

def binomial_lln(sample_size, p):

    ## we are sampling from a $B(1,p)$ distribution
    ##

    ######
    ## Step 1 - create sample of independent uniform random variables

    n = sample_size
    lower_bound = 0.
    upper_bound = 1.
    uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

    ######
    ## Step 2 - transform them to $B(1,p)$ distribution

    sample = [0.] * n
    for j in range(n):
        sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

    x_ax = [0.] * n  # values on the x axis
    n_plots = 2
    y_ax = [0.] * n_plots  # values on the x axis
    # y_values (0) - running average
    y_ax[0] = [0.] * n
    # y_values (1) - actual average
    y_ax[1] = [p for x in range(n)]

    ######
    ## we want to plot the cumulative average of all the samples
    sum = sample[0]
    y_ax[0][1] = sum
    for k in range(1, n):
        x_ax[k] = k
        sum = sum + sample[k-1]
        y_ax[0][k] = sum / (k+1)

    mp = pu.PlotUtilities("Cumulative Average", 'x', 'Average')
    mp.multiPlot(x_ax, y_ax)

def binomial_lln_hist(sample_size, repeats, p):

    ##
    ## we are sampling from a $B(1,p)$ distribution
    ##

    n = sample_size  # this is how often we sample each time.

    lower_bound = 0.
    upper_bound = 1.

    plotCLN = True
    sample_value = range(repeats)

    for i in sample_value:

        ## Step 1 - create sample of independent uniform random variables

        ### This code uses uniform random variables and the quantile transform
        uni_sample = np.random.uniform(lower_bound, upper_bound, sample_size)

        sample = [0.] * n
        for j in range(n):
            sample[j] = dist.binomial_inverse_cdf(p,uni_sample[j])

        sum = sample[0]
        for k in range(1, n):
            sum = sum + sample[k - 1]

        if plotCLN:
            sample_value[i] = (sum - sample_size * p) / np.sqrt(sample_size * p * (1-p))
        else:
            sample_value[i] = sum / (sample_size)

        ## sample value for CLN


    ##
    ## we then turn the outcome into a histogram
    ##
    num_bins = 35

    # the histogram of the data
    _n, bins, _hist = plt.hist(sample_value, num_bins, normed=True, facecolor='green', alpha=0.75)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')
    plt.title("Histogram of Average Sample of Size={1} of B(1,p) with p={0}".format(p, sample_size))

    ###### overlay the actual normal distribution
    if plotCLN:

        y = range(0,num_bins+1)
        for m in range(0,num_bins+1):
             y[m] = dist.standard_normal_pdf(bins[m])

        plt.plot(bins, y, 'r--')

    plt.show()

    #######

if __name__ == '__main__':

    sz = 1500
    p = .75
#    binomial_lln(sz, p)

    repeats = 10000
    binomial_lln_hist(sz, repeats, p)


