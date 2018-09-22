################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import matplotlib.pyplot as plt

class PlotUtilities():

    def __init__(self, title, x_label, y_label):

        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    ###############
    ##
    ##  utility to plot histogram
    ##
    ###############

    def plotHistogram(self, sample_data, num_bins):

        plt.hist(sample_data, num_bins, normed=True, facecolor='green', alpha=0.75)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)

        plt.show()

    ###############
    ##
    ##  utility to plot multiple histograms
    ##
    ###############

    def plotMultiHistogram(self, samples, num_bins, colors = 'None'):

        n_plots = len(samples)

        base_alpha = 0.55
        for k in range(n_plots):
            # the histogram of the data
            _thisAlpha = base_alpha + 0.25 * float(k)
            if (colors == 'None'):
                plt.hist(samples[k], num_bins, normed=True, facecolor='blue', alpha=_thisAlpha)
            else:
                plt.hist(samples[k], num_bins, normed=True, facecolor=colors[k], alpha=_thisAlpha)

        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)

        plt.show()

    ###############
    ##
    ##  utility to plot multiple graphs at once
    ##
    ###############

    def multiPlot(self, x_ax, y_ax, arg = ''):

        #######
        ## sizing of axis
        #######
        n_plots = len(y_ax)
        t_min = 100
        t_max = -100.
        for k in range(n_plots):
            t_min = min(t_min, min(y_ax[k]))
            t_max = max(t_max, max(y_ax[k]))

        ########
        ## some basic formatting
        ########
        plt.xlabel(self.x_label)
        plt.ylabel(self.y_label)
        plt.title(self.title)
        plt.axis([min(x_ax), max(x_ax), 0.9 * t_min, 1.1 * t_max])

        ########
        ## actual plotting
        ########
        for k in range(n_plots):
            plt.plot(x_ax, y_ax[k], arg)

        plt.show()

