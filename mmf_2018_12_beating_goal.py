################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################


import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def plot_maximising_goal_probability(_time, _timestep, _initial_capital, _target, _b, _r, _sigma):

    #######
    ## call helper function to generate sufficient symmetric binomials
    #######

    size = int(_time / _timestep) - 1

    sample = np.random.normal(0, np.sqrt(_timestep), size)

    path_underlying = [1.] * (size)
    path_wealth = [_initial_capital] * (size)
    path_portfolio = [0] * (size)

    x = [0.0] * (size)

    for k in range(size):
        x[k] = _timestep * k

    _theta = (_b - _r) / _sigma

    _y0 = np.sqrt(_time) * dist.standard_normal_inverse_cdf(_initial_capital * np.exp(_r * _time) / _target)

    ####
    ## create the various paths for plotting
    ####

    bm = 0
    _y = path_wealth[0] * np.exp(_r * _time) / _target
    path_portfolio[0] = dist.standard_normal_pdf(dist.standard_normal_inverse_cdf(_y)) / (_y * _sigma * np.sqrt(_time))
    for j in range(1, size):
        _t_remain = _time - x[j]
        _t_sq_remain = np.sqrt(_t_remain)
        path_underlying[j] = path_underlying[j-1] * (1. + _b * _timestep + _sigma * sample[j])
        bm = bm + sample[j] + _theta * _timestep
        path_wealth[j] = _target * np.exp( - _r * _t_remain ) * \
                                dist.standard_normal_cdf((bm + _y0) / _t_sq_remain)

        _y = path_wealth[j] * np.exp(_r * _t_remain) / _target
        path_portfolio[j] = dist.standard_normal_pdf(dist.standard_normal_inverse_cdf(_y)) / (_y * _sigma * _t_sq_remain)


    mp = pu.PlotUtilities("Maximising Probability of Reaching a Goal", 'Time', "None")

    y_ax = [0] * 3
    y_ax[0] = path_underlying
    y_ax[1] = path_wealth
    y_ax[2] = path_portfolio

    labels = ['Stock Price', 'Wealth Process', 'Portfolio Value']
    mp.subPlots(x, y_ax, labels, ['red',
                                  'blue', 'green'])



if __name__ == '__main__':

    _initial_capital = 1
    _target_wealth = 1.20

    _time = 2.
    timestep = 0.001

    _b = 0.08
    _r = 0.05
    _sigma = .30

    plot_maximising_goal_probability(_time, timestep, _initial_capital, _target_wealth, _b, _r, _sigma)
