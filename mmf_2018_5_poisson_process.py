################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu

def poisson_process(intensity, compensate):

    ## we are sampling the first sz jumps
    sz = 100

    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, sz)

    #######
    ### transform the uniform sample to exponentials
    #######

    sample = [0.] * sz
    for j in range(sz):
        sample[j] = dist.exponential_inverse_cdf(intensity,uni_sample[j])

    jumps = [0.] * sz
    for j in range(sz):
        if (j == 0):
            jumps[j] = sample[j]
        else:
            jumps[j] = jumps[j-1] + sample[j]

    ####
    ## plot the trajectory of the process
    ####

    steps = 1000
    step_size = 0.05
    y = [0.] * 1
    x = [0] * steps
    y[0] = [0] * steps

    for k in range(steps):
        x[k] = k * step_size
        for l in range(sz):
            if (jumps[l] > x[k]):
                y[0][k] = l
                break
        if compensate:
            y[0][k] = y[0][k] - intensity * x[k]


    #######
    ### prepare and show plot
    ###
    mp = pu.PlotUtilities("Trajectory of Poisson Process with Intensity ={0}".format(intensity), 'Time', '# Of Jumps')
    mp.multiPlot(x, y)

if __name__ == '__main__':

    intensity = 0.25
    compensate = True
    poisson_process(intensity, compensate)


