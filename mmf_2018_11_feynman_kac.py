################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

# from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt
import numpy as np

import core_math_utilities as dist

##
## Example of Feynman Kac Equation
##

##
## Goal is that we want to plot functions u(t,x) where
## the function solves the equation
## u_t + Au - ru = 0
## In these cases we know what particular form the function has
## namely \mathb{E}( f(X(t)) | X(0) = x)
## and $X$ being the process for which Au is the infinitesimal generator
##

##
## Examples of different functors for both the
## Geometric Brownian Motion and BM with Drift
##

class Functor:
    def u(self, _t, _x):
        return 0.

    def z_max(self):
        return 1.0


class FunctorTransitionDensity(Functor):
    def __init__(self, _start):
        self._start = _start

    def u(self, _t, _x):
        nd = dist.NormalDistribution(_x, _t)
        return nd.pdf(self._start)


class FunctorEPE(Functor):
    def u(self, _t, _x):
        nd = dist.NormalDistribution(_x, _t)
        return nd.expected_positive_exposure()


class Functor2ndMoment(Functor):
    def u(self, _t, _x):
        nd = dist.NormalDistribution(_x, _t)
        return nd.second_moment()

    def z_max(self):
        return 10.0


class FunctorExcessProbability(Functor):
    def __init__(self, _strike):
        self._strike = _strike

    def u(self, _t, _x):
        nd = dist.NormalDistribution(_x, _t)
        return nd.excess_probability(self._strike)


class FunctorExpectedValueGBM(Functor):
    def __init__(self, a, b, k):
        self.a = a
        self.b = b
        self.exponent = k

    def u(self, _t, _x):
        k = self.exponent
        return np.exp(k * self.a * _t + 0.5 * k * (k - 1) * self.b * self.b * _t) * np.power(_x, k)

    def z_max(self):
        return 5.0


class FunctorExcessProbabilityGBM(Functor):
    def __init__(self, a, b, strike):
        self.a = a
        self.b = b
        self.strike = strike

    def u(self, _t, _x):
        mean = (self.a - 0.5 * self.b * self.b) * _t
        variance = self.b * self.b * _t
        nd = dist.NormalDistribution(mean, variance)
        strk = np.log(self.strike / _x)
        return 1 - nd.cdf(strk)


def plot_3d(functor2D):
    ### set up proportions of the figure
    fig = plt.figure(figsize=plt.figaspect(0.6))

    # set up the axes for the first plot
    ax = fig.gca(projection='3d')
    X = np.arange(0.1, 3, 0.2)
    Y = np.arange(0.01, 6, 0.05)
    # Y = np.arange(-3.0, 3., 0.05)
    X, Y = np.meshgrid(X, Y)
    Z = functor2D.u(X, Y)
    z_lim = functor2D.z_max()

    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm,
                           linewidth=0, antialiased=True)

    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_zlim(0, z_lim)
    fig.colorbar(surf, shrink=0.75, aspect=8)

    plt.show()


if __name__ == '__main__':
    f_epe = FunctorEPE()
    f_tp = FunctorTransitionDensity(0.)
    f_2m = Functor2ndMoment()
    f_expr = FunctorExcessProbability(-0.5)

    f_gbm_exp = FunctorExpectedValueGBM(-0.25, 0.0, 1.)
    f_gbm_exprob = FunctorExcessProbabilityGBM(-0.25, 0.25, 2.)

    f = f_gbm_exp
    plot_3d(f)
