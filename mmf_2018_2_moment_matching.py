################
## Author: Thomas Balzer
## (c) 2018
## Material for MMF Stochastic Analysis - Fall 2018
################

import numpy as np
import matplotlib.pyplot as plt

import core_math_utilities as dist

###########
##
## Demo of Moment Matching
##
## - Sample a basket of a given size with random weights between (0,1)
## - Those basket constituents are assumed to follow a binomial distribution on $\{-1,+1\}$ with success probability $p$.
## - We then sample the outcome of the basket and compare it to a normal distribution with matching moments.
## - We can look at the distribution itself but also at tail probabilities
##
###########

def moment_matching(p, sz_basket):

    #####
    ## create a basket with random weights of size sz_basket
    #####

    lower_bound = 0.
    upper_bound = 1.

    weights = np.random.uniform(lower_bound, upper_bound, sz_basket)

    ## calculate mean and variance of the basket

    expectation = 0
    variance = 0

    for k in range(sz_basket):
        expectation = expectation + weights[k]
        variance = variance + weights[k] * weights[k]

    expectation = expectation * dist.symmetric_binomial_expectation(p)
    variance = variance * dist.symmetric_binomial_variance(p)

    simulation = 50000

    outcome = [0] * simulation

    for k in range(simulation):

        uni_sample = np.random.uniform(lower_bound, upper_bound, sz_basket)

        #######
        ### transform the uniform sample
        #######
        sample = [0.] * sz_basket
        for j in range(sz_basket):
            sample[j] = dist.symmetric_binomial_inverse_cdf(p, uni_sample[j])

        for m in range(sz_basket):
            outcome[k] = outcome[k] + weights[m] * sample[m]

    num_bins = 50


    plt.subplot(2,1,1)

    plt.title("Moment Matching of Binomial Basket of Size={0}".format(sz_basket))

    # the histogram of the data
    n, bins, _hist = plt.hist(outcome, num_bins, normed=True, facecolor='blue', alpha=0.75)

    pdf_approx = [0.] * (num_bins+1)
    call_option_sampled = [0.] * (num_bins+1)
    call_option_approx = [0.] * (num_bins+1)

    nd = dist.NormalDistribution(expectation, variance)

    ###### overlay the moment matched pdf
    for i in range(0,num_bins+1):
        pdf_approx[i] = nd.pdf(bins[i])
        call_option_approx[i] = nd.call_option(bins[i])
        call_option_sampled[i] = 0.
        for k in range(simulation):
            if (outcome[k] >= bins[i]):
                call_option_sampled[i] = call_option_sampled[i] + (outcome[k] - bins[i]) / float(simulation)

    plt.xlabel('Outcome')
    plt.ylabel('Rel. Occurrence')

    plt.plot(bins, pdf_approx, 'r*')

    plt.subplot(2,1,2)

    plt.xlabel('Call Option Strike')
    plt.ylabel('Call Option Price')

    plt.plot(bins, call_option_sampled, 'b-')
    plt.plot(bins, call_option_approx, 'r*')

    plt.show()



if __name__ == '__main__':

    moment_matching(0.75, 15)

