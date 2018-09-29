################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################


import matplotlib.pyplot as plt
import numpy as np

import plot_utilities as pu

def bivariate_normal_scatter(mu_1, mu_2, sigma_1, sigma_2, rho):
    ### scatter plot of bivariate normal distribution
    ### we sample 2 * sizes from a standard normal distribution and then create a correlated
    ###

    size = 500
    standard_normal_sample = np.random.standard_normal(2 * size)
    ### we turn this into a random sample of dimension 2;

    x = [0.] * (size)

    n_plots = len(rho)
    y = [0.] * n_plots
    for k in range(n_plots):
        y[k] = [0.] * (size)

    mp = pu.PlotUtilities('Bivariate Normal Distribution with Varying Correlations'.format(rho), 'x', 'y')

    for k in range(size):
        x[k] = sigma_1 * standard_normal_sample[k] + mu_1
        for m in range(n_plots):
            z = rho[m] * standard_normal_sample[k] + np.sqrt(1 - rho[m] * rho[m]) * standard_normal_sample[k + size]
            y[m][k] = sigma_2 * z + mu_2

    mp.scatterPlot(x, y, rho)


if __name__ == '__main__':

    mu = 0
    sigma_sq = 2.5
#    rho = -0.9
    rho = [-0.9, 0., 0.9]
    bivariate_normal_scatter(mu, mu, sigma_sq, sigma_sq, rho)

