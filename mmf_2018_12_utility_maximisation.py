################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import plot_utilities as pu

def terminal_utility_histogram(_b, _r, _sigma, T, _sample_size):

    #####
    ## plot the terminal utility of a stock vs an optimal strategy (for various utility functions)
    #####

    n = _sample_size  # this is how often we sample each time.

    sample = np.random.normal(0, T, n)

    alpha = .0
    sample_value_stock = [0.] * n
    sample_value_pi = [0.] * n
    pi = (_b - _r) / (_sigma * _sigma) / (1 - alpha)
    sigma_pi = _sigma * pi
    b_pi = _r + pi * (_b - _r)
    for i in range(n):
        normal_sample = sample[i]
        stock_value = np.exp((_b + 0.5 * _sigma * _sigma) * T + _sigma * 1. * normal_sample)
        sample_portfolio = np.exp((b_pi + 0.5 * sigma_pi * sigma_pi) * T + sigma_pi * 1. * normal_sample)
        sample_value_stock[i] = stock_value
        sample_value_pi[i] = sample_portfolio


    samples = [0] * 2
    samples[0] = sample_value_stock
    samples[1] = sample_value_pi

    labels = ['Stock', 'Portfolio']

    ##
    ## we then turn the outcome into a histogram
    ##

    num_bins = 100

    mp = pu.PlotUtilities("Terminal Wealth for Stock and Mixed Portfolio for $\pi=${0}".format(pi), 'Outcome', 'Rel. Occurrence')
    mp.plotHistogram(samples, num_bins, labels)




if __name__ == '__main__':
    _b = .2
    _sigma = 0.5
    _r = 0.05
    _t = 1.
    _n = 25000

    terminal_utility_histogram(_b, _r, _sigma, _t, _n)

