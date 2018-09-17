################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


###############
##
##  Plot Normal Distribution CDF vs Chernoff Bound
##
###############

def plotNormal(min_val, max_val, var, steps):

        step = (max_val - min_val) / steps
        n_plots = 3

        nd = dist.NormalDistribution(0., var)

        # values on the x axis
        x_ax = [0.] * steps
        for k in range(steps):
            x_ax[k] = min_val + step * k

        ## container for y axis
        y_ax = [0.] * n_plots
        for k in range(n_plots):
            y_ax[k] = [0.] * steps

        for k in range(steps):
            y_ax[0][k] = nd.excess_probability(x_ax[k])
            y_ax[1][k] = np.exp(-0.5 * x_ax[k] * x_ax[k] / var)
            y_ax[2][k] = 0.5 * var / (x_ax[k] * x_ax[k])

        mp = pu.PlotUtilities('Chernoff Bound for $N(0,\sigma^2)$ with $\sigma^2$={0}'.format(var), 'x', 'Probability')
        mp.multiPlot(x_ax, y_ax)


###############
##
##  Plot Exponential CDF vs Various Bounds
##
###############

def plotExponential(min_val, max_val, _lambda, steps):

        step = (max_val - min_val) / steps
        n_plots = 4

        xd = dist.ExponentialDistribution(_lambda)

        # values on the x axis
        x_ax = [0.] * steps
        for k in range(steps):
            x_ax[k] = min_val + step * k

        ## container for y axis
        y_ax = [0.] * n_plots
        for k in range(n_plots):
            y_ax[k] = [0.] * steps

        for k in range(steps):
            _x = x_ax[k]
            y_ax[0][k] = xd.excess_probability(_x)
            y_ax[1][k] = 1. / (_x * _lambda)  ## \mathbb{P}( X > a ) \leq \mathbb{E} / a - Simple Markov Bound
            y_ax[2][k] = 2. / (_x * _x * _lambda * _lambda)  ## \mathbb{P}( X > a ) \leq \mathbb{E^2} / a^2 - Chebychef Bound
            y_ax[3][k] = np.exp(-_lambda * _x) * np.exp(1.) * _x * _lambda

        mp = pu.PlotUtilities('Probability Bounds for $Exp(\lambda)$ with $\lambda$={0}'.format(_lambda), 'x', 'Probability')
        mp.multiPlot(x_ax, y_ax)




if __name__ == '__main__':

    min_val = 0.25
    max_val = 0.999999
    steps = 100
    var = .5
    plotExponential(min_val, max_val, var, steps)

