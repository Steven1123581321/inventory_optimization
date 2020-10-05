from config import get_config
import numpy as np
import pandas as pd
import math
import scipy.stats as st
from scipy.stats import poisson
import sympy
from sympy import Symbol, diff
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

##########################################
#BISECTION ALGORITHM#
#Deterministic EOQ with waste#
##########################################

def bisection_example():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    algorithm_parameters = get_config("algorithm_parameters", "bisection")
    p = cost_parameters["buying_price"]
    w = cost_parameters["waste_cost"]
    h = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    x_m = demand_parameters["demand_during_lifetime"]
    d = demand_parameters["yearly_demand"]
    f = lambda x: p*h*0.5-((d*o)/(x-(x*poisson.pmf(x_m,x,loc=0)+(x-x_m)*(1-poisson.cdf(x_m,x,loc=0))))**2)*(1-poisson.cdf(x,x_m,loc=0))+w*d*((-1*(x*poisson.pmf(x_m,x,loc=0)+(x-x_m)*(1-poisson.cdf(x_m,x,loc=0)))/x**2)+(poisson.cdf(x,x_m,loc=0)/x))
    EOQ_no_waste = np.sqrt((2*d*o)/(h*p))
    EOQ_with_waste = bisection(f,1,EOQ_no_waste,algorithm_parameters["number_of_iterations"])
    quantities = []
    costs = []
    for Q in range(1, int(EOQ_no_waste)):
        F_x = poisson.cdf(x_m,Q,loc=0)
        EO = (Q*poisson.pmf(x_m,Q,loc=0)+(Q-x_m)*(1-F_x))
        IC = h*p*Q/2
        BC = (d*o)/(Q-EO)
        WC = EO*w*(d/Q)
        cost = IC+BC+WC
        quantities.append(Q)
        costs.append(cost)
    if algorithm_parameters["plot_waste_algorithm_outcome"]:
        plt.plot(costs)
        plt.ylabel("Order quantity")
        plt.xlabel("Costs")
        plt.title("The optimal order quantity is "+str(EOQ_with_waste))
        plt.show()
    else:
        print(" The optimal order quantity is ", EOQ_with_waste)

def bisection(f,a,b,N):
    if f(a)*f(b) >= 0:
        return None
    a_n = a
    b_n = b
    for n in range(1,N+1):
        m_n = (a_n + b_n)/2
        f_m_n = f(m_n)
        if f(a_n)*f_m_n < 0:
            a_n = a_n
            b_n = m_n
        elif f(b_n)*f_m_n < 0:
            a_n = m_n
            b_n = b_n
        elif f_m_n == 0:
            return m_n
        else:
            return None
    return (a_n + b_n)/2

##########################################
#Hooke-Reeves ALGORITHM#
#Stochastic EOQ with geometric demand#
##########################################

def hooke_reeves_example():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    algorithm_parameters = get_config("algorithm_parameters", "hooke_reeves")
    p = cost_parameters["buying_price"]
    i = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    d = demand_parameters["yearly_demand"]
    h = i*p
    b = demand_parameters["beta_value_geometric_distribution"]
    f = lambda x: ((o+p*x[0]+h*((x[0]*(x[0]-1))/2))/(1/(1-b)+x[0]))+(h*x[0])/(1+(x[0]*(1-b)))
    solution = hooke_reeves(f, [50], 1, 0.01, [1])
    for i in range(len(solution)):
        print("x"+str(i)+"=", solution[i])

def hooke_reeves(f, x, a, eps, minimum, fraction=0.5):
    y, n = f(x), len(x)
    while a > eps:
        improved=False
        x_best, y_best = x, y
        for i in range(n):
            for i in range(-1, 2):
                x_prime = x+np.eye(n)*i*a
                x_prime = np.diag(x_prime)
                y_prime = f(x_prime)
                if y_prime < y_best:
                    x_best, y_best, improved = x_prime, y_prime, True
        x, y = x_best, y_best
        if not improved:
            a *= fraction
        for i in range(len(x)):
            if x[i] <= minimum[i]:
                return minimum
    return x

##########################################
#RMSProp algorithmM#
##function x_0**2+x_1**2##
##########################################

'''
do the example with the EOQ. Now you need to change the learning rate to 0.9.
This shows the parametrization signficance for neural networks optimization.
'''

def RMSProp_example():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    algorithm_parameters = get_config("algorithm_parameters", "adagrad")
    p = cost_parameters["buying_price"]
    h = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    d = demand_parameters["yearly_demand"]
    '''
    x = Symbol('x')
    y = Symbol('y')
    f = x**2+y**2
    partialderiv= diff(f, x)
    partial-derivative_1 = partialderiv.doit()
    partialderiv= diff(f, y)
    partial-derivative_2 = partialderiv.doit()
    '''
    p_1 = lambda x: 2*x[0]
    p_2 = lambda x: 2*x[1]
    solution = np.round(RMSProp(p_1, p_2, [50,50], 0.01, 0.1, 0.9, 10000),2)
    for i in range(len(solution)):
        print("x"+str(i)+"=", solution[i])

def RMSProp(p_1,p_2, x, a, e, gamma, n):
    r= len(x)
    for k in range(n):
        for i in range(r):
            if i == 0:
                x_prime = p_1(x)
            if i == 1:
                x_prime = p_2(x)
            s = x_prime**2
            if k==0:
                RMS = math.sqrt(s)
            else:
                RMS = math.sqrt(gamma*s_prime+(1-gamma)*s)
            x[i] -= (a*x_prime)/(e+RMS)
            s_prime=s
    return x


##########################################
#Metropolis-Hastings algorithm#
##forecasting##
##########################################
def get_proposal(mean_current, std_current, proposal_width = 0.5):
        return np.random.normal(mean_current, proposal_width), np.random.normal(std_current, proposal_width)

def accept_proposal(mean_proposal, std_proposal, mean_current, std_current, prior_mean, prior_std, data):
    def prior(mean, std, prior_mean, prior_std):
        return st.norm(prior_mean[0], prior_mean[1]).logpdf(mean)+ st.norm(prior_std[0], prior_std[1]).logpdf(std)

    def likelihood(mean, std, data):
        return np.sum(st.norm(mean, std).logpdf(data))

    prior_current = prior(mean_current, std_current, prior_mean, prior_std)
    likelihood_current = likelihood(mean_current, std_current, data)
    prior_proposal = prior(mean_proposal, std_proposal, prior_mean, prior_std)
    likelihood_proposal = likelihood(mean_proposal, std_proposal, data)
    return (prior_proposal + likelihood_proposal) - (prior_current + likelihood_current)

def get_trace(data, samples = 5000):
    mean_prior = 5
    std_prior = 5
    mean_current = mean_prior
    std_current = std_prior
    trace = {
        "mean": [mean_current],
        "std": [std_current]
    }
    for i in range(samples):
        mean_proposal, std_proposal = get_proposal(mean_current, std_current)
        acceptance_prob = accept_proposal(mean_proposal, std_proposal, mean_current, \
                                         std_current, [mean_prior, std_prior], \
                                          [mean_prior, std_prior], data)
        if math.log(np.random.rand()) < acceptance_prob:
            mean_current = mean_proposal
            std_current = std_proposal
        trace['mean'].append(mean_current)
        trace['std'].append(std_current)
    return trace

def metropolis_hasting_example():
    meanX = 1.5
    stdX = 1.2
    X = np.random.normal(meanX, stdX, size = 1000)
    trace = get_trace(X)
    mean = np.mean(trace['mean'])
    std = np.mean(trace['std'])
    print(mean, std)

##########################################
#Automatic differentiation#
##partial derivatives of x**2+y**2##
##########################################

class DualNumber:
    def __init__(self, real, dual):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real + other.real,
                              self.dual + other.dual)
        else:
            return DualNumber(self.real + other, self.dual)

    def __sub__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real - other.real,
                              self.dual - other.dual)
        else:
            return DualNumber(self.real - other, self.dual)

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real,
                              self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)
    __rmul__ = __mul__

    def __truediv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real/other.real,
                              (self.dual*other.real - self.real*other.dual)/(other.real**2))
        else:
            return (1/other) * self

    def __floordiv__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real/other.real,
                                (self.dual*other.real - self.real*other.dual)/(other.real**2))
        else:
            return (1/other) * self

    def __rtruediv__(self, other):
        return DualNumber(other, 0).__truediv__(self)

    def __rfloordiv__(self, other):
        return DualNumber(other, 0).__floordiv__(self)

    def __pow__(self, other):
        return DualNumber(self.real**other,
                          self.dual * other * self.real**(other - 1))

    def __repr__(self):
        return repr(self.real) + ' + ' + repr(self.dual) + '*epsilon'

def auto_diff(f, x):
    return f(DualNumber(x, 1.)).dual

def diff():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    p = cost_parameters["buying_price"]
    h = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    d = demand_parameters["yearly_demand"]
    p1 = auto_diff(lambda x: d/x*o+x/2*h*p, 17 )
    x = 1
    p2 = auto_diff(lambda y: x**2+y**-1, 5 )
    print(p1, p2)

##########################################
#Simulated Annealing#
##Auchkley function##
##########################################


##########################################
#Point iteration algorithm#
##Nonstocked decision function##
##########################################

##########################################
#Cuckoo Search#
##Nonstocked decision function##
##########################################