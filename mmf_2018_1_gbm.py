################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np
import plot_utilities as pu

###############
##
##  simple comparison of account value for different compounding frequencies
##
###############

def compounding_plot(rate, freq, min_val, max_val, steps):


    step = (max_val - min_val) / steps
    n_plots = len(freq)

    # values on the x axis
    x_ax = [0.] * steps
    for k in range(steps):
        x_ax[k] = min_val + step * k

    ## container for y axis
    y_ax = [0.] * n_plots
    for k in range(n_plots):
        y_ax[k] = [0.] * steps

    starting_value = 1.
    for k in range(n_plots):
        #######
        ### linear interpolation on a grid given by the compounding frequency
        #######
        __n = int((max_val - min_val) / freq[k]) + 1
        x_temp = [min_val] * __n
        y_temp = [starting_value] * __n
        for m in range(1, __n):
            x_temp[m] = min_val + float(m) * freq[k]
            y_temp[m] = y_temp[m-1] * (1. + rate * freq[k])

        y_ax[k] = np.interp(x_ax, x_temp, y_temp)

    mp = pu.PlotUtilities('Compounding Account Value', 'x', 'Value')
    mp.multiPlot(x_ax, y_ax)

###############
##
##  adding a random shock to a fixed growth rate at any point
##
###############

def distorted_plot(rate, vols, min_val, max_val, steps):

    step = (max_val - min_val) / steps
    n_plots = len(vols)

    # values on the x axis
    x_ax = [0.] * steps
    for k in range(steps):
        x_ax[k] = min_val + step * k

    ## container for y axis
    starting_value = 1.
    y_ax = [0.] * n_plots
    for k in range(n_plots):
        y_ax[k] = [starting_value] * steps

    for k in range(n_plots):
        for m in range(1, steps):
            if (vols[k] <= 0. ):
                random_shock = 0.
            else:
                random_shock = np.random.normal(0, vols[k] * np.sqrt(step), 1)
            y_ax[k][m] = y_ax[k][m-1] * (1. + rate * step + random_shock)

    mp = pu.PlotUtilities('Compounding Account Value With Noise', 'x', 'Value')
    mp.multiPlot(x_ax, y_ax)

if __name__ == '__main__':

    rate = 0.1
    freq = [5, 0.5, 0.05, 0.01]
    min_val = 0.
    max_val = 10.
    steps = 100
 #   compounding_plot(rate, freq, min_val, max_val, steps)

    rate = 0.1
#    vols = [0., 0.01, 0.05, 0.1]
    vols = [0.1] * 5
    vols[0] = 0.
    min_val = 0.
    max_val = 10.
    steps = 1000
    distorted_plot(rate, vols, min_val, max_val, steps)

