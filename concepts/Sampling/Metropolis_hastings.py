import numpy as np
import scipy.stats
import math

def prior_probability(beta):
    a = beta[0]
    b = beta[1]
    a_prior = scipy.stats.norm(0.5, 0.5).pdf(a)
    b_prior = scipy.stats.norm(0.5, 0.5).pdf(b)
    return np.log(a) + np.log(b)

def likelihood_probability(beta, x, y, e):
    a = beta[0]
    b = beta[1]
    y_predict = a+b*x
    return -np.log(math.sqrt(2*math.pi*e**2))-(1/(2*e**2))*np.sum((y-y_predict)**2)

def posterior_probability(beta, x, y, e):
    return likelihood_probability(beta, x, y, e) + prior_probability(beta)

def proposal_function(beta):
    a = beta[0]
    b = beta[1]
    a_new = np.random.normal(a, 0.5)
    b_new = np.random.normal(b, 0.5)
    beta_new = [a_new, b_new]
    return beta_new