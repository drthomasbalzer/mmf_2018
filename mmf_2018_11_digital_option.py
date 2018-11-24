################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def plot_bachelier_digital_option(_time, _timestep, _strike):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    size = int(_time / _timestep)

    sample = np.random.normal(0, np.sqrt(_timestep), size)

    path_bm = [0] * (size)
    path_digital_option = [0] * (size)
    hedge_proportion = [0] * (size)
    path_digital_option_hedge = [0] * (size)

    x = [0.0] * (size)

    for k in range(size):
        x[k] = _timestep * k

    ####
    ## plot the trajectory of the process
    ####
    _t_remain = np.sqrt(_time - x[0])
    path_bm[0] = 0.
    path_digital_option[0] = 1 - dist.standard_normal_cdf((_strike - path_bm[0]) / _t_remain)
    path_digital_option_hedge[0] = path_digital_option[0]
    hedge_proportion[0] = dist.standard_normal_pdf((_strike - path_bm[0]) / _t_remain) / _t_remain

    # for j in range(1, size):
    #     if (j == 0):
    #         path_bm[j] = 0.
    #         path_digital_option[j] = 1 - dist.standard_normal_cdf((_strike - path_bm[j]) / _t_remain)
    #         path_digital_option_hedge[j] = path_digital_option[j]
    #     else:
    #         path_bm[j] = path_bm[j - 1] + sample[j]
    #         path_digital_option_hedge[j] = path_digital_option_hedge[j - 1] + sample[j] * dist.standard_normal_pdf(
    #         (_strike - path_bm[j - 1]) / _t_remain) / _t_remain
    #         path_digital_option[j] = 1 - dist.standard_normal_cdf((_strike - path_bm[j]) / _t_remain)

    for j in range(1, size):
        path_bm[j] = path_bm[j - 1] + sample[j]
        hedge_proportion[j] = dist.standard_normal_pdf((_strike - path_bm[j - 1]) / _t_remain) / _t_remain
        path_digital_option_hedge[j] = path_digital_option_hedge[j-1] + sample[j] * hedge_proportion[j]
        path_digital_option[j] = 1 - dist.standard_normal_cdf((_strike - path_bm[j]) / _t_remain)

    mp = pu.PlotUtilities("Paths of Digital Option Value", 'Time', "Option Value")

    trackHedgeOnly = True
    if (trackHedgeOnly):
        y = [0] * 2
        y[0] = path_digital_option
        y[1] = path_digital_option_hedge
        mp.multiPlot(x, y)
    else:
        y = [0] * 3
        y[0] = path_digital_option
        y[1] = hedge_proportion
        y[2] = path_bm

        arg = ['Option Value', 'Hedge Proportion', 'Underlying Brownian Motion']
        colors = ['green', 'red', 'blue']
        mp = pu.PlotUtilities("Paths of Digital Option Value", 'Time', "Option Value")
        mp.subPlots(x, y, arg, colors)


if __name__ == '__main__':

    time = .5
    timestep = 0.0005
    strike = 1.
    plot_bachelier_digital_option(time, timestep, strike)
