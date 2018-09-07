import matplotlib.pyplot as plt

###############
##
##  utility to plot multiple graphs at once
##
###############

class MultiPlot():

    def __init__(self, title, x_label, y_label):

        self.title = title
        self.x_label = x_label
        self.y_label = y_label

    def plot(self, x_ax, y_ax):

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
            plt.plot(x_ax, y_ax[k])

        plt.show()

