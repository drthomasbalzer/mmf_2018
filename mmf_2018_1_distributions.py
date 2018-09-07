import plot_utilities as pu
import core_math_utilities as dist

###############
##
##  set up plots for pdf of multiple distributions
##
###############

def plotMultiDistributions(distrib, min_val, max_val, steps):

        step = (max_val - min_val) / steps
        n_plots = len(distrib)

        # values on the x axis
        x_ax = [0.] * steps
        for k in range(steps):
            x_ax[k] = min_val + step * k

        ## container for y axis
        y_ax = [0.] * n_plots
        for k in range(n_plots):
            y_ax[k] = [0.] * steps

        for j in range(n_plots):
            for k in range(steps):
                y_ax[j][k] = distrib[j].pdf(x_ax[k])

        mp = pu.MultiPlot('Probability Density Functions', 'x', 'PDF Value')
        mp.plot(x_ax, y_ax)

if __name__ == '__main__':

    m = 5
    d = [0.] * m

    version = 1

    if (version == 1):
        ### example of normal distributions with different $\sigma$ and $\mu = 0$.
        min_val = -5.
        max_val = 5.
        for k in range(m):
            d[k] = dist.NormalDistribution(0., 1. * (1. + float(k)) )
    if (version == 2):
        min_val = -5.
        max_val = 5.
        ### example of normal distributions with different $\mu$ and $\sigma = 1$.
        for k in range(m):
            d[k] = dist.NormalDistribution(-2. + float(k), 1.)
    if (version == 3):
        ## example of Exponential distributions
        min_val = 0.
        max_val = 4.
        for k in range(m):
            d[k] = dist.ExponentialDistribution(0.5 * (1. + float(k)))

    plotMultiDistributions(d, min_val, max_val, 100)

