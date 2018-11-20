################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import plot_utilities as pu

#############
##  -- Generic Representation of an Ito Process
##  --- of the form dX(t) = (A(t)X(t) + a(t)) dt + (D(t) X(t) + d(t)) dB(t)
#############

class itoProcessGeneral:

   def drift(self, _t, _x):
       return 0.

   def diffusion(self, _t, _x):
       return 0.

class itoProcessTH(itoProcessGeneral):

   def __init__(self, _A, _a, _D, _d, _x):

       self._A = _A
       self._a = _a
       self._D = _D
       self._d = _d

       self.initial_value = _x

   def drift(self, _t, _x):
       return self._A * _x + self._a

   def diffusion(self, _t, _x):
       return self._D * _x + self._d

class itoProcessBMDrift(itoProcessTH):

   def __init__(self, _a, _d, _x):

       self._A = 0.
       self._a = _a
       self._D = 0.
       self._d = _d

       self.initial_value = _x

class itoProcessGBM(itoProcessTH):

   def __init__(self, _A, _D, _x):

       self._A = _A
       self._a = 0.
       self._D = _D
       self._d = 0.

       self.initial_value = _x

class itoProcessBrownianBridge(itoProcessGeneral):

   def __init__(self, _T, _b, _sigma, _x):

       self._T = _T
       self._b = _b
       self._sigma = _sigma

       self.initial_value = _x

   def drift(self, _t, _x):
       return (self._b - _x) / (self._T - _t)

   def diffusion(self, _t, _x):
       return self._sigma



def plot_sde(_maxTime, _timestep, _number_paths, itoPr):

   #######
   ## call helper function to generate sufficient symmetric binomials
   #######

   ## normals for a single paths

   size = int(_maxTime / _timestep)

   ## total random numbers needed
   total_sz = size * _number_paths

   sample = np.random.normal(0, np.sqrt(_timestep), total_sz)

   paths = [0.] * (_number_paths)

   x = [0.0] * (size + 1)

   for k in range(size + 1):
       x[k] = _timestep * k

   ####
   ## plot the trajectory of the Ito process
   ####
   i = 0
   for k in range(_number_paths):
       process = 0
       path = [1] * (size + 1)
       for j in range(size + 1):
           if (j == 0):
               process = itoPr.initial_value
               path[j] = process
               continue ## nothing
           else:

               ########
               ## the paths will be constructed through application of Ito's formula
               ########

               _x = process
               _u = x[j-1]
               _du = _timestep
               _dBu = sample[i]
               _a = itoPr.drift(_u, _x)
               _b = itoPr.diffusion(_u, _x)

               ####
               ## -- underlying ito process
               ## X(t + dt) = X(t) + a(t,x) dt + b(t,x) dB(t)
               ####

               process = process + _a * _du + _b * sample[i]
               path[j] = process

               ### increment counter for samples

               i = i + 1

       paths[k] = path

   #######
   ### prepare and show plot
   ###
   mp = pu.PlotUtilities('Paths of Ito Process', 'Time', 'Value')
   mp.multiPlot(x, paths)


if __name__ == '__main__':

   max_time = 5
   timestep = 0.001
   paths = 12

   ### ito process of the form dX = (A X(t) + a) dt + (B X(t) + b) dB(t)

   ### ito process of the form dX = a dt + b dB(t)
   ito_bm = itoProcessBMDrift(.40, .2, 0)

   ### ito process of the form dX = X(a dt + b dB(t))
   ito_exp = itoProcessGBM(0., 0.2, 1)

   ### ito process of the form dX = X(t) dt + b dB(t)
   ito_1 = itoProcessTH(0.3, 0.0, 0., 0.5, 1)

   ### ito process of the form dX = mean X(t) dt + b dB(t)
   ito_2 = itoProcessTH(-0.05, 0.0, 0., 0.05, 1)

   ### ito process of the form dX = mean X(t) dt + b dB(t)
   ##
   ito_mr = itoProcessTH(-.5, 1, 0., 0.5, 10)

   ##
   ### Brownian Bridge
   ##
   ito_bb = itoProcessBrownianBridge(max_time, 1., .75, 0.)

   ito = ito_mr


   plot_sde(max_time, timestep, paths, ito)