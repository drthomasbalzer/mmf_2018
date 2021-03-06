################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################


import numpy as np
import plot_utilities as pu

def uniform_histogram_powers(sz, powers):

    lower_bound = 0.
    upper_bound = 1.

    n_samples = len(powers) + 1
    samples = [0.] * n_samples

    for k in range(n_samples):
        samples[k] = [0.] * sz

    samples[0] = np.random.uniform(lower_bound, upper_bound, sz)

    labels = [1.] * n_samples
    for k in range(len(powers)):
        labels[k+1] = powers[k]
    num_bins = 25

    for k in range(sz):
        for l in range(len(powers)):
            samples[l+1][k] = np.power(samples[0][k], powers[l])

    mp = pu.PlotUtilities("Histogram of Uniform Sample of Size={0}".format(sz), 'Outcome', 'Rel. Occurrence')
    mp.plotHistogram(samples, num_bins, labels)

if __name__ == '__main__':

    sz = 100000
    powers = [1.25, 1.5, 2.]
    uniform_histogram_powers(sz, powers)


