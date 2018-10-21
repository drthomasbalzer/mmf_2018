################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np

import core_math_utilities as dist
import plot_utilities as pu


def malliavin_greeks(start, vol, strike):

    ## we calculate the option value of a call option $(X-K)^+$ where the underlying is of the form $X = x_0 + sigma W$ with $W$ standard normal
    ## the aim is to calculate the sensitivities of the option price with respect to the x_0 both in bump and reval and with logarithmic Malliavin weights

    nd = dist.NormalDistribution(start, vol * vol)
    theo_option_price = nd.call_option(strike)

    perturbation = 0.00000001
    #    nd_pert = dist.NormalDistribution(start + perturbation, vol * vol)
    #    theo_option_price_pert = nd_pert.call_option(strike)
    #    theo_delta = (theo_option_price_pert - theo_option_price) / perturbation

    #    nd_pert_down = dist.NormalDistribution(start - perturbation, vol * vol)
    #    theo_option_price_pert_down = nd_pert_down.call_option(strike)
    #    theo_gamma = (theo_option_price_pert - 2. * theo_option_price + theo_option_price_pert_down) / perturbation / perturbation

    y = (start - strike) / vol
    act_delta = dist.standard_normal_cdf(y)  # - y * dist.standard_normal_pdf(y)
    act_gamma = dist.standard_normal_pdf(y) / vol
    print (str("Theoretical Price: ") + str(theo_option_price))
    print (str("Theoretical Delta: ") + str(act_delta))
    print (str("Theoretical Gamma: ") + str(act_gamma))

    repeats = 500

    sample_delta = [0.] * 2
    sample_delta[0] = [0] * repeats  # this is the sample for the delta with B&R approach
    sample_delta[1] = [0] * repeats  # this is the sample for the delta with Malliavin logarithmic trick
    #    sample_delta[2] = [0] * repeats # this is the sample for the delta with B&R approach
    #    sample_delta[3] = [0] * repeats # this is the sample for the delta with Malliavin logarithmic

    sz = 5000
    total_sz = sz * repeats

    normal_sample = np.random.normal(0, 1, total_sz)
    s_count = 0

    for z in range(repeats):

        option_value = 0.
        option_value_pert = 0.
        option_value_pert_down = 0.
        option_delta = 0.
        option_gamma = 0.
        for j in range(sz):
            x = start + vol * normal_sample[s_count]
            option_value = option_value + max(x - strike, 0.)

            x_pert = x + perturbation
            option_value_pert = option_value_pert + max(x_pert - strike, 0.)

            x_pert_down = x - perturbation
            option_value_pert_down = option_value_pert_down + max(x_pert_down - strike, 0.)

            log_weight_1 = (x - start) / (vol * vol)
            log_weight_2 = ((x - start) * (x - start) / (vol * vol) - 1.) / (vol * vol)
            #          log_weight = (x - start) * (x - start) / (vol * vol * vol) - 1./vol

            option_delta = option_delta + max(x - strike, 0.) * log_weight_1
            option_gamma = option_gamma + max(x - strike, 0.) * log_weight_2
            s_count = s_count + 1

        # sample_delta[0][z] = (option_value_pert - option_value) / sz / perturbation
        sample_delta[0][z] = (
                             option_value_pert - 2 * option_value + option_value_pert_down) / sz / perturbation / perturbation
        # sample_delta[1][z] = option_delta / sz
        sample_delta[1][z] = option_gamma / sz
    #######
    ### prepare and show plot
    ###
    num_bins = 25

    print (np.var(sample_delta[0]))
    print (np.var(sample_delta[1]))

    colors = ['#0059ff', '#db46e2', '#ffc800', '#99e45e']
    _title = "Option Delta with B&R vs Malliavin Greeks"

    mp = pu.PlotUtilities(_title, 'Outcome', 'Rel. Occurrence')
    mp.plotMultiHistogram(sample_delta, num_bins, colors)


if __name__ == '__main__':
    start = 5.
    strike = 7.5
    vol = 1.0
    #    start = np.log(100.)
    #    strike = 10.
    #    vol = .25
    malliavin_greeks(start, vol, strike)