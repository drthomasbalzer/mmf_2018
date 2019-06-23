################
## Author: Thomas Balzer
## (c) 2019
################

import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist
import plot_utilities as pu


def sample_uniforms():
    size = 50000
    lower_bound = 0.
    upper_bound = 1.

    uni_sample = np.random.uniform(lower_bound, upper_bound, size)
    return uni_sample


def example_lognormal(mean, variance):
    uni_sample = sample_uniforms()
    nd = dist.NormalDistribution(0, variance)
    #######
    ### transform the uniform sample
    #######
    ###
    sample = [mean * np.exp(nd.inverse_cdf(u)) for u in uni_sample]
    num_bins = 100

    plot_title = "Histogram of Lognormal Sample with Mean={0}, Variance={1}".format(mean, variance)
    x_label = 'Outcome'
    y_label = 'Rel. Occurrence'
    hp = pu.PlotUtilities(plot_title, x_label, y_label)
    hp.plotHistogram(sample, num_bins)


def example_lognormal_with_payoff(mean, variance, strike):
    uni_sample = sample_uniforms()
    nd = dist.NormalDistribution(0, variance)

    sample = [mean * np.exp(nd.inverse_cdf(u)) for u in uni_sample]
    num_bins = 100

    plot_title_global = 'Option Payoff'
    plot_title = "Histogram of Lognormal Sample with Mean={0}, Variance={1}".format(mean, variance)
    x_label = 'Outcome'
    y_label = 'Rel. Occurrence'

    plt.title(plot_title_global)

    plt.subplot(2, 1, 1)
    plt.title(plot_title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    n, bins, _hist = plt.hist(sample, num_bins, normed=True, alpha=0.75, color='#dc450f')

    plt.subplot(2, 1, 2)
    y = [max(bins[k] - strike, 0) for k in range(0, num_bins + 1)]
    plt.xlabel('Option Payoff vs Strike={0}'.format(strike))
    plt.plot(bins, y, 'r')
    plt.subplots_adjust(left=0.15)
    plt.show()
    plt.close()


def example_lognormal_path(min_val, max_val, steps, mean, variance, rate, includeRiskFree=True):
    step = (max_val - min_val) / steps
    # values on the x axis
    x_ax = [min_val + step * k for k in range(steps)]
    vol = np.sqrt(variance)

    n_plots = 12
    ## container for y axis
    y_ax = [[mean for j in range(steps)] for k in range(n_plots)]

    for k in range(n_plots):
        for m in range(1, steps):
            if (k == 0 and includeRiskFree):
                random_shock = 0.
            else:
                random_shock = np.random.normal(0, vol * np.sqrt(step), 1)
            y_ax[k][m] = y_ax[k][m - 1] * np.exp(rate * step + random_shock - 0.5 * vol * vol * step)

    if (includeRiskFree):
        mp = pu.PlotUtilities('Stock Price Realisations Vs Risk-Free Investment', 'x', 'Value')
    else:
        mp = pu.PlotUtilities('Stock Price Realisations', 'x', 'Value')
    mp.multiPlot(x_ax, y_ax)


def bs_formula_call(stock, strike, time, vol, rate):
    d_1 = (np.log(stock / strike) + (rate + 0.5 * vol * vol) * time) / (vol * np.sqrt(time))
    d_2 = d_1 - vol * np.sqrt(time)

    n_d_1 = dist.standard_normal_cdf(d_1)
    bs_price = n_d_1 * stock - dist.standard_normal_cdf(d_2) * np.exp(-rate * time) * strike

    return bs_price, n_d_1


def example_option_hedging(_time, _timestep, _initial_value, _variance, _rate, _strike):
    size = int(_time / _timestep) - 1
    #    print (size)

    sample = np.random.normal(0, np.sqrt(_timestep), size)
    vol = np.sqrt(_variance)
    path_underlying = [_initial_value] * (size)
    path_wealth = [0.] * (size)
    path_portfolio = [0.] * (size)

    x = [_timestep * k for k in range(size)]

    ####
    ## create the various paths for plotting
    ####
    for j in range(size):
        if (j == 0):
            path_underlying[j] = _initial_value
        else:
            path_underlying[j] = path_underlying[j - 1] * np.exp(
                _rate * _timestep + vol * sample[j] - 0.5 * _timestep * _variance)

        s = path_underlying[j]
        _t_remain = _time - x[j]
        path_wealth[j], d_1 = bs_formula_call(s, strike, _t_remain, vol, rate)
        path_portfolio[j] = dist.standard_normal_cdf(d_1)

    # indx = size-1
    #    print (path_underlying[indx])
    #    print (path_wealth[indx])
    mp = pu.PlotUtilities("Hedging of European Stock Option", 'Time', "None")

    y_ax = [0] * 3
    y_ax[0] = path_underlying
    y_ax[1] = path_wealth
    y_ax[2] = path_portfolio

    labels = ['Stock Price', 'Option Value', 'Hedge Ratio']
    mp.subPlots(x, y_ax, labels, ['red',
                                  'blue', 'green'])


def example_option_gap_risk(_time, _timestep, _initial_value, _variance, _rate, _strike, hazard_rate, gap_horizon,
                            includeEnvelope, includeCapital=False):
    size = int(_time / _timestep) - 1

    sample = np.random.normal(0, np.sqrt(_timestep), size)
    vol = np.sqrt(_variance)
    path_underlying = [_initial_value] * (size)
    path_wealth = [0.] * (size)
    path_gap = [0.] * (size)
    gap_envelope = [0.] * size
    expected_loss = [0.] * size
    gap_el = [0.] * size

    x = [_timestep * k for k in range(size)]

    percentile = 0.99
    inv_q = dist.normal_CDF_inverse(percentile)
    perc_time = _timestep * gap_horizon
    perc_env = np.exp(vol * np.sqrt(perc_time) * inv_q - 0.5 * perc_time * _variance)
    ####
    ## create the various paths for plotting
    ####
    for j in range(size):
        if (j == 0):
            path_underlying[j] = _initial_value
        else:
            path_underlying[j] = path_underlying[j - 1] * np.exp(
                _rate * _timestep + vol * sample[j] - 0.5 * _timestep * _variance)

        s = path_underlying[j]
        _t_remain = _time - x[j]
        path_wealth[j], d_1 = bs_formula_call(s, strike, _t_remain, vol, rate)
        pd = 1. - np.exp(- hazard_rate * _t_remain)
        expected_loss[j] = pd * path_wealth[j]
        if (j >= gap_horizon):
            path_gap[j] = path_wealth[j] - path_wealth[j - gap_horizon]
            gap_el[j] = expected_loss[j] - expected_loss[j - gap_horizon]
        gap_envelope[j] = d_1 * (perc_env - 1) * s

    # indx = size-1
    #    print (path_underlying[indx])
    #    print (path_wealth[indx])
    mp = pu.PlotUtilities("European Stock Option - Gap Risk Analysis", 'Time', "None")

    y_ax = [0] * 3
    y_ax[0] = path_underlying
    y_ax[1] = path_wealth
    y_ax[2] = path_gap

    if (includeEnvelope):
        mp = pu.PlotUtilities('Worst Case Move Envelope for 10-Day Gap Risk', 'Time', 'Moves in Value')
        mp.multiPlot(x, [path_gap, gap_envelope])
    elif (includeCapital):
        y_ax[2] = expected_loss
        y_ax.append(gap_el)
        mp = pu.PlotUtilities('Expected Loss and CVA {0}-Day Gap Risk'.format(gap_horizon), 'Time', 'Moves in Value')
        labels = ['Stock Price', 'Option Value', 'Expected Loss', 'CVA Gap ']
        mp.subPlots(x, y_ax, labels, ['red',
                                      'blue', 'green', 'orange'])
    else:
        labels = ['Stock Price', 'Option Value', '{0}-Day Move'.format(gap_horizon), 'Worst Case Move']
        mp.subPlots(x, y_ax, labels, ['red',
                                      'blue', 'green', 'orange'])


def example_bivariate_scatter(mean, variance, pd, pd_vol, rho):
    size = 5000
    standard_normal_sample = np.random.standard_normal(2 * size)
    ### we turn this into a random sample of dimension 2;

    vol = np.sqrt(variance)
    x = [mean * np.exp(vol * standard_normal_sample[k] - 0.5 * variance) for k in range(size)]
    y = [0. for k in range(size)]

    #    colors = ['blue', 'green', 'orange', 'red', 'yellow']
    colors = ['#ff5012', 'red', 'yellow']
    mp = pu.PlotUtilities('Bivariate Distribution of Stock and Default with Correlation={0}'.format(rho),
                          'Default Indicator', 'Stock Value')

    default_threshold = dist.normal_CDF_inverse(pd)
    for k in range(size):
        z = pd_vol * (rho * standard_normal_sample[k] + np.sqrt(1 - rho * rho) * standard_normal_sample[k + size])
        if z <= default_threshold:
            y[k] = 0.  ### zero indicates default
        else:
            y[k] = 1.  ### one indicates survival

    mp.scatterPlot(y, [x], [rho], colors)


def example_bivariate_option_price_mc(mean, variance, pd, pd_vol, strike, df):
    size = 8000
    standard_normal_sample = np.random.standard_normal(2 * size)
    ### we turn this into a random sample of dimension 2;

    vol = np.sqrt(variance)
    x = [mean * np.exp(vol * standard_normal_sample[k] - 0.5 * variance) for k in range(size)]
    y = [0. for k in range(size)]

    option_value = df * sum([max(x_0 - strike, 0) for x_0 in x]) / size
    min_rho = -0.99
    max_rho = 0.99
    step_size = 0.01
    rho_steps = int((max_rho - min_rho) / step_size)
    rhos = [min_rho + k * step_size for k in range(rho_steps)]
    option_values = [option_value for rho in rhos]

    mc_value = []
    default_threshold = dist.normal_CDF_inverse(pd)
    for rho in rhos:
        for k in range(size):
            z = pd_vol * (rho * standard_normal_sample[k] + np.sqrt(1 - rho * rho) * standard_normal_sample[k + size])
            if z <= default_threshold:
                y[k] = 0
            else:
                y[k] = 1.
        mc_value.append(df * sum([y_0 * max(x_0 - strike, 0.) for y_0, x_0 in zip(y, x)]) / size)

    mp = pu.PlotUtilities('Risk-Adjusted Option Value As Function of Correlation', 'Correlation', 'Option Value')
    mp.multiPlot(rhos, [option_values, mc_value])


if __name__ == '__main__':

    #### examples in order
    ## 0. lognormal distribution of underlying
    ## 1. lognormal distribution of underlying vs payoff of option
    ## 2. path of stock price
    ## 3. path of a stock price vs a risk free underlying
    ## 4. path of stock price vs value function vs hedge ratio
    ## 5. scatter plot - stock price and default indicator, no correlation
    ## 6. scatter plot - stock price and default indicator, negative correlation
    ## 7. scatter plot - stock price and default indicator, positive correlation
    ## 8. bivariate option price as a function of correlation
    ## 9. 10-day moves of option price over time
    ## 10. Envelope of 10-day moves of option price over time
    ## 11. Expected loss and CVA gap over time

    #    example_number = 8

    # examples = [k for k in range(12)]
    examples = [11]
    current_value = 100
    mult = 1.05
    df = 1. / mult
    rate = np.log(mult)
    #    print (rate)
    time = 1.
    forward_value = current_value * np.exp(time * rate)
    variance = 0.04
    strike = 90
    time_step = 1. / 365

    default_prob = 0.1
    pd_vol = 1.0
    hazard_rate = -np.log((1. - default_prob) / time)
    
    for example_number in examples:
        if (example_number == 0):  ### basic histogram
            example_lognormal(forward_value, variance)
        elif (example_number == 1):  ### basic histogram
            example_lognormal_with_payoff(forward_value, variance, strike)
        elif (example_number == 2):  ### basic histogram not including the rf rate
            example_lognormal_path(0., 1., 10000, current_value, variance, rate, False)
        elif (example_number == 3):  ### basic histogram not including the rf rate
            example_lognormal_path(0., 1., 10000, current_value, variance, rate, True)
        elif (example_number == 4):
            ### hedge ratio over time of european stock option
            example_option_hedging(time, time_step, current_value, variance, rate, strike)
        elif (example_number == 5):  ### bivariate distribution of stock and default - no correlation
            example_bivariate_scatter(forward_value, variance, default_prob, pd_vol, 0.0)
        elif (example_number == 6):  ### bivariate distribution of stock and default - negative correlation
            example_bivariate_scatter(forward_value, variance, default_prob, pd_vol, -0.75)
        elif (example_number == 7):  ### bivariate distribution of stock and default - positive correlation
            example_bivariate_scatter(forward_value, variance, default_prob, pd_vol, +0.75)
        elif (example_number == 8):  ### bivariate distribution of stock and default - positive correlation
            example_bivariate_option_price_mc(forward_value, variance, default_prob, pd_vol, strike, df)
        elif (example_number == 9):  ### 10-day moves of option price
            example_option_gap_risk(time, time_step, current_value, variance, rate, strike, hazard_rate, 10, False)
        elif (example_number == 10):  ### 10-day moves of option price
            example_option_gap_risk(time, time_step, current_value, variance, rate, strike, hazard_rate, 10, True)
        elif (example_number == 11):  ### 10-day moves of option price
            example_option_gap_risk(time, time_step, current_value, variance, rate, strike, hazard_rate, 10, False,
                                    True)

