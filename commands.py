from config import get_config
import sys
from warnings import catch_warnings
from warnings import simplefilter
from collections import defaultdict
from concepts.Automatic_differentiation.reverse_accumulation import reverse_autodiff, Overloader
from concepts.service_levels.optimizer import Optimizer
from concepts.interview.question_class import Graph
from concepts.Gradient_boosting.regression_tree import regression_tree
from sklearn.datasets import load_boston
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
import autograd.numpy as numpy
import time
from autograd import grad
import numpy as np
from numpy import log, exp, pi
import pandas as pd
import random
import math
import scipy.stats as st
from scipy.special import gamma
from scipy.stats import poisson, cauchy, norm, uniform
import statsmodels.api as sm
import sympy
from sympy import Symbol, diff
import matplotlib.pyplot as plt
from pyomo.environ import *
from pyomo.opt import SolverFactory
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
##function Rosenbrock and eoq formula##
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
    p_1 = lambda x: -2*(1-x[0])-4*5*x[0]*(x[1]-x[0]**2)
    p_2 = lambda x: 2*5*(x[1]-x[0]**2)
    p_3 = lambda x:-(d*o)/x[0]**2+h*p*.5
    # add p_2 when having partial derivatives.
    solution = np.round(RMSProp(p_1, p_2, [10,10], 0.01, 0.1, 0.9, 2000000),2)
    for i in range(len(solution)):
        print("x"+str(i)+"=", solution[i])

# Add p_2 when having partial derivatives.
def RMSProp(p_1, p_2, x, a, e, gamma, n):
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
#Adam algorithmM#
##function -(d*o)/x[0]**2+h*p*.5##
##########################################

def adam_example():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    algorithm_parameters = get_config("algorithm_parameters", "adagrad")
    p = cost_parameters["buying_price"]
    h = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    d = demand_parameters["yearly_demand"]
    f = lambda x:-(d*o)/x**2+h*p*.5
    solution = adam(f, 100, 0.9, 0.999, 0.000000001, 0.001, 2000000)
    print("x"+"=", solution)

def adam(f, x, gamma_v, gamma_s, e, a, n):
    for k in range(n):
        x_prime = f(x)
        x_hat = x
        if k == 0:
            v_prime = 0
            s_prime = 0
        else:
            v_prime = gamma_v*v+(1-gamma_v)*x_prime
            s_prime = gamma_s*s+(1-gamma_s)*x_prime**2
        x = x_hat-a*v_prime/(e+math.sqrt(s_prime))
        s = s_prime
        v = v_prime
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

def get_trace(data, samples = 500):
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
    X = np.random.normal(meanX, stdX, size = 100)
    trace = get_trace(X)
    mean = np.mean(trace['mean'])
    std = np.mean(trace['std'])
    print(mean, std)


##########################################
#Metropolis-Hastings algorithm#
##http://www.mit.edu/~ilkery/papers/MetropolisHastingsSampling.pdf##
##########################################
def get_trace1(x, y, samples = 10000):
    rho = 0
    # Store the samples
    chain_rho=numpy.array([0.]*samples)
    p_current = rho
    N = 1000
    proposal_width = 0.07
    for i in range(samples):
        rho_proposal  = np.random.uniform(p_current-proposal_width ,p_current+proposal_width)
        # Compute the acceptance probability, Equation 8 and Equation 6. 
        # We will do both equations in log domain here to avoid underflow.
        accept = -3./2*log(1.-rho_proposal**2) - N*log((1.-rho_proposal**2)**(1./2)) - sum(1./(2.*(1.-rho_proposal**2))*(x**2-2.*rho_proposal*x*y+y**2))
        accept = accept-(-3./2*log(1.-p_current**2) - N*log((1.-p_current**2)**(1./2)) - sum(1./(2.*(1.-p_current**2))*(x**2-2.*p_current*x*y+y**2)))
        accept = min([1,accept])
        accept = exp(accept)
        if uniform.rvs(0,1) < accept:
            p_current = rho_proposal
        chain_rho[i]=p_current
    return chain_rho

def metropolis_hasting_example2():
    N=1000
    data=np.random.multivariate_normal([0,0],[[1, 0.4],[0.4, 1]],N)
    x=data[:,0]
    y=data[:,1]
    trace = get_trace1(x, y)
    f, (ax1,ax2,ax3)=plt.subplots(3,1)
    # Plot the data
    ax1.scatter(x,y,s=20,c='b',marker='o')
    # plot things
    ax2.plot(trace,'b')
    ax2.set_ylabel('$rho$')
    ax3.hist(trace,50)
    ax3.set_xlabel('$rho$')

    plt.show()

##########################################
#Metropolis-Hastings algorithm#
##geometric distributed demand and looking for the distribution of beta##
##pagina 82 Axsäter##

# wanneer we de formule op p 82 van axsater toepassen dan moeten we veel data hebben#
#bv bij N = 1000, dan zitten we er heel kort bij#
# maar bij N = 100, is beta rond de 0,20-0,25#
# met het algoritme kunnen we direct zien dat het gemiddelde 0,35 is en zien we zelfs een distributie ervan#
# hoe groter de sample, hoe kleiner de afwijking en hoe zekerder dat we zijn#
# dit is niet alleen heel handig bij het bepalen van het order level bij weinig transacties,
# maar ook outlier detection wanneer er weinig transacties zijn#


##FORMULES: zie rode boekje en document MHAforbook.docx!!!#####
##########################################
def get_trace2(x, start=0.01, N = 100, samples = 100000):
    beta = start
    # Store the samples
    beta_chain = numpy.array([0.]*samples)
    beta_current = beta
    for i in range(samples):
        beta_proposal  = np.random.uniform(0.01 , 0.99)
        # Compute the acceptance probability.
        # We will do both equations in log domain here to avoid underflow.
        accept = (N*log(beta_proposal)+sum(x-1)*log(1-beta_proposal))
        accept = accept-(N*log(beta_current)+sum(x-1)*log(1-beta_current))
        accept = min([1,accept])
        accept = exp(accept)
        if uniform.rvs(0,1) < accept:
            beta_current = beta_proposal
        beta_chain[i]=beta_current
    return beta_chain

def metropolis_hasting_example3():
    N=100
    data=np.random.geometric(0.60,N)
    x = data
    beta_from_mean = 1/x.mean()
    trace = get_trace2(x)
    f, (ax1,ax2)=plt.subplots(2,1)
    # plot things
    ax1.plot(trace,'b')
    ax1.set_ylabel('$beta$')
    ax2.hist(trace, 50)
    ax2.set_xlabel('$beta$')
    plt.show()

##########################################
#Automatic differentiation- forward accumulation#
##derivative of d/x*o+x/2*h*p##
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
    __rsub__=__sub__

    def __mul__(self, other):
        if isinstance(other, DualNumber):
            return DualNumber(self.real * other.real,
                              self.real * other.dual + self.dual * other.real)
        else:
            return DualNumber(self.real * other, self.dual * other)
    __rmul__ = __mul__

    def __pow__(self, other):
        return DualNumber(self.real**other,
                          self.dual * other * self.real**(other - 1))

def auto_diff(f, x):
    return f(DualNumber(x, 1.)).dual

def forward_diff():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    p = cost_parameters["buying_price"]
    h = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    d = demand_parameters["yearly_demand"]
    f = auto_diff(lambda x: (1-x)**2+5*(10-x**2)**2, 1)
    print(f)

##########################################
#Automatic differentiation: reverse accumulation#
##derivative of d/x*o+x/2*h*p##
##########################################

def reverse_diff():
    cost_parameters = get_config("cost_parameters")
    demand_parameters = get_config("demand_parameters")
    p = cost_parameters["buying_price"]
    h = cost_parameters["holding_cost_percentage"]
    o = cost_parameters["order_cost"]
    d = demand_parameters["yearly_demand"]
    x = Overloader(245)
    f = reverse_autodiff(d/x*o+x/2*h*p, x)
    print(f)


##########################################
#Adjusted RMSE error#
##########################################

def rmse_adjusted_2():
    df = pd.read_excel("C:\\Users\\s.pauly\\Documents\\Business Cases\\Autonet\\Covariance4.xlsx")
    def autocv(df):
        autocovariances = []
        variance_sum = []
        df = df.iloc[:,1:]
        for i in range(len(df)):
            x = np.array(df.iloc[i,:])
            autocov_matrix = sm.tsa.stattools.acovf(x)
            autocov_sum = autocov_matrix[1:].sum()
            autocovariances.append(autocov_sum)
            sum_variance = np.std(x)**2*len(df.columns)
            variance_sum.append(sum_variance)
        return autocovariances,variance_sum
    df['covariance_correction'],df['variance_sum']= autocv(df)
    df['corrected_sum'] = df['variance_sum']+2*df['covariance_correction']
    df['normal_rmse'] = np.sqrt(df['variance_sum']/len(df.iloc[:,1:-3].columns))
    df['adjusted_rmse'] = np.sqrt(df['corrected_sum']/len(df.iloc[:,1:-4].columns))
    print(df)

##########################################
#calculating impact centralization#
##########################################

def rmse_adjusted():
    df = pd.read_excel("C:\\Users\\s.pauly\\Documents\\Business Cases\\Autonet\\Covariance4.xlsx")
    df = df.iloc[:,1:]
    data = np.array(df)
    covMatrix = np.cov(data,bias=True)
    # centralized_variance = sum(sum(covMatrix))
    # decentralized_variance = sum(np.diagonal(covMatrix))
    decentralized_ss = sum(np.sqrt(np.diagonal(covMatrix)))
    centralized_ss = np.sqrt(sum(sum(covMatrix)))
    difference_in_ss = decentralized_ss-centralized_ss
    print(difference_in_ss)

##########################################
#Simulated Annealing#
##Ackley function##
##########################################

def simulated_annealing_example():
    a = 20
    b = 0.2
    c = 2 * np.pi
    f = lambda x: (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2
    # ackley: a + np.exp(1) -a * np.exp(-b * np.sqrt(x[0]*x[0] + x[1]*x[1]) / 2) -np.exp((np.cos(c*x[0]) + np.cos(c*x[1]))/2)
    # Beale : (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2
    # 3*(1-x[0])**2*math.exp(-x[0]**2-(x[1]+1)**2)-10*(x[0]/5-x[0]**3-x[1]**5)*math.exp(-x[0]**2-x[1]**2)-1/3*math.exp(-(x[0]+1)**2-x[1]**2)
    # f = lambda x: (a-x[0])**2+b*(x[1]-x[0]**2)**2
    solution = simulated_annealing(f, [-2,2], 2, 100, 100000, .99)
    for i in range(len(solution)):
        print("x"+str(i)+"=", solution[i])

def simulated_annealing(f, x, T, t, k_max, gamma):
    r = len(x)
    y = f(x)
    x_best, y_best = x, y
    for i in range(k_max):
        p = [random.uniform(-T, T) for i in range(r)] 
        x_prime = [x + y for x, y in zip(x, p)]
        y_prime = f(x_prime)
        y_delta = y_prime-y
        if y_delta <= 0 or np.random.rand() < math.exp(-y_delta/t):
            x, y = x_prime, y_prime
        if y_prime<y_best:
            x_best, y_best = x_prime, y_prime
        t *= gamma
    return x_best


##########################################
#Point iteration algorithm#
##Nonstocked decision function##
##########################################

def fixed_point_iteration():
    mu = 1000
    h = 10
    s = 2.05
    T = 1
    Q = 10
    o = 15
    f = lambda x: x*(norm.ppf(1-(Q*h)/(x*mu)) *s*math.sqrt(T)*h+math.sqrt(2*mu*o*h))/(mu*x-h*Q)
    solution = point_iteration(f)
    print("the margin boundary is", solution, "€")

def point_iteration(f, k_max = 100, start = 50):
    m = start
    counter = 0
    while counter <= k_max:
        g_M = f(m)
        m = g_M
        counter += 1
    return m

##########################################
#Cuckoo Search#
##########################################

def cuckoo_search_example():
    factor = lambda x: ((gamma(1+x)*math.sin(math.pi*x*0.5))/(gamma((1+x)/2)*x*2**((x-1)/2)))**(1/x)
    f = lambda x: (1.5-x[0]+x[0]*x[1])**2+(2.25-x[0]+x[0]*x[1]**2)**2+(2.625-x[0]+x[0]*x[1]**3)**2
    solution, min_value = cuckoo_search(f, factor)
    for i in range(len(solution)):
        print("x"+str(i)+"=", solution[i])

def cuckoo_search(f, factor, range_x = [-4.5, 4.5], population=50, d=0.01, beta = 1.5, p_a = 0.25, k_max = 10000, dimensions = 2):
    # Initialize an initial nest.
    initial_nest = np.random.rand(population, dimensions)*(range_x[1]-(range_x[0]))+range_x[0]
    # Calculate factor sigma_u.
    sigma_u = factor(beta)
    counter = 0
    while counter <=k_max:
        if counter == 0:
            nest = initial_nest
            values  = [f(initial_nest[egg]) for egg in range(len(initial_nest))]
        best_egg = nest[values.index(min(values))]
        # Calculate u, v and s.
        for i in range(len(nest)):
            u = np.random.normal(0,sigma_u, dimensions)
            v = np.random.normal(0,1, dimensions)
            s = u/(abs(v)**(1/beta))
            new_values = []
            # Calculate new solutions/eggs with the lévy flight.
            for j in range(dimensions):
                x_new = nest[i, j]+ np.random.standard_cauchy()*d*s[j]*(nest[i, j]-best_egg[j])
                new_values.append(x_new)
            new_objective_value = f(new_values)
            # If new objective value of new egg is lower than the current egg, replace it in the nest.
            if new_objective_value < values[i]:
                values[i]=new_objective_value
                nest[i]=new_values  
            new_values_2= []
            r = np.random.rand(dimensions)
            d_1 = random.uniform(1, population)
            d_2 = random.uniform(1, population)
            for j in range(dimensions):
                if r[j] < p_a:
                    # random.rand() is a uniform distribution. This is so much worse than a gauchy!
                    x_new = nest[i, j]+ np.random.standard_cauchy()*(nest[int(d_1), j]-nest[int(d_2), j])
                    new_values_2.append(x_new)
                else:
                    x_new = nest[i, j]
                    new_values_2.append(x_new)
            new_objective_value = f(new_values_2)
            if new_objective_value < values[i]:
                values[i]=new_objective_value
                nest[i]=new_values
        print(min(values))
        counter += 1
    return nest[values.index(min(values))], min(values)

##########################################
#Gradient Boosting Algorithm#
##########################################

def gradient_boosting_example():
    tree = regression_tree()
    tree.load('boston')
    tree.prepare_dataset()
    M = 10
    learning_rate = 0.10
    f_0 = np.mean(tree.y_train)
    output_data = pd.DataFrame(tree.y_train)
    input_data = pd.DataFrame(tree.X_train)
    output_data['f_x'] = f_0
    for i in range(M):
        # Put the derivative as a new output column.
        output_data['r_im']  = output_data['House Price']-output_data['f_x']
        # best_variable_split = tree.split(input_data, output_data['r_im'])
        decision_tree = DecisionTreeRegressor()
        decision_tree.fit(input_data, output_data['r_im'])
        output_data['R_jm'] = decision_tree.apply(input_data)
        leafs = output_data['R_jm'].unique().tolist()
        output_data['gamma_jm'] = 0.
        for leaf in leafs:
            array = output_data['House Price'].loc[output_data['R_jm']==leaf]
            average = np.mean(array)
            output_data['gamma_jm'] = np.where(output_data['R_jm']==leaf, average, output_data['gamma_jm'])
        output_data['update_value'] = output_data['gamma_jm'] - output_data['f_x']
        output_data['f_x'] = output_data['f_x'] + learning_rate*output_data['update_value']
        output_data = output_data[['House Price', 'f_x']]
    return output_data['f_x']

def gradient_boosting_quantile_regression_example():
    tree = regression_tree()
    tree.load('boston')
    tree.prepare_dataset()
    M = 100
    Quantile = 0.75
    learning_rate = 0.01
    f_0 = np.quantile(tree.y_train, Quantile)
    output_data = pd.DataFrame(tree.y_train)
    input_data = pd.DataFrame(tree.X_train)
    output_data['f_x'] = f_0
    decision_trees = []
    for i in range(M):
        # Put the derivative as a new output column.
        output_data['r_im']  = -((output_data['House Price']-output_data['f_x'] < 0) - Quantile)
        # best_variable_split = tree.split(input_data, output_data['r_im'])
        decision_tree = DecisionTreeRegressor()
        decision_tree.fit(input_data, output_data['r_im'])
        decision_trees.append(decision_tree)
        output_data['R_jm'] = decision_tree.apply(input_data)
        leafs = output_data['R_jm'].unique().tolist()
        output_data['gamma_jm'] = f_0
        for leaf in leafs:
            array = output_data['House Price'].loc[output_data['R_jm']==leaf]
            inverse_quantile = np.quantile(array, Quantile)
            output_data['gamma_jm'] = np.where(output_data['R_jm']==leaf, inverse_quantile, output_data['gamma_jm'])
        output_data['update_value'] = output_data['gamma_jm'] - output_data['f_x']
        output_data['f_x'] = output_data['f_x'] + learning_rate*output_data['update_value']
        output_data = output_data[['House Price', 'f_x']]
    def gradient_boost_mse_predict(regressors, f0, X, learning_rate):
        y_hat = np.array([f0]*len(X)) 
        for regressor in regressors: 
            y_hat += learning_rate * regressor.predict(X)
        return y_hat
    y_hat = gradient_boost_mse_predict(decision_trees, f0=f_0, X = tree.X_test, learning_rate = learning_rate)
    truth = tree.y_test
    # truth = tree.y_train
    # y_hat = output_data['f_x'].values
    correct = 0.
    for i, val in enumerate(truth):
        if val <= y_hat[i]:
            correct += 1
    print(correct/len(truth))
    return output_data['f_x']

##########################################
#Simple search algorithm for Smoothing constant#
##########################################

def mean_squared_error(observations, predictions):
    squared_error = (observations-predictions)**2
    return np.mean(squared_error)

def search(data, start=0.5, samples = 50, starting_value=None):
    # Store the samples
    data = pd.DataFrame(data)
    alpha_chain = numpy.array([0.]*samples)
    alpha_current = start
    proposal_width = 0.111111111111
    error_current = np.sum(data**2)
    error_current = error_current.values[0]
    for i in range(samples):
        alpha_proposal  = np.random.uniform(alpha_current-proposal_width ,alpha_current+proposal_width)
        if alpha_proposal >1.:
            alpha_proposal = 1.
        elif alpha_proposal < 0.:
            alpha_proposal = 0.
        else:
            alpha_proposal = alpha_proposal
        # Compute the acceptance probability.
        x = []
        prediction = starting_value
        x.append(prediction)
        for j in range(len(data)-1):
            prediction = data.iloc[j,:]*alpha_proposal + (1-alpha_proposal)*prediction
            x.append(prediction.values[0])
        data['predictions'] = x
        error_proposal = mean_squared_error(data['Demand'], data['predictions'])
        accept = error_proposal/error_current
        accept = min([1,accept])
        if 1 > accept:
            alpha_current = alpha_proposal
            error_current = error_proposal
        alpha_chain[i]=alpha_current
    return alpha_chain

def simple_search_example():
    dataset = pd.read_excel("./output/Book1.xlsx")
    data = dataset['Demand']
    N = 100
    startTime = time.time()
    starting_value = np.mean(data[:-N])
    data = data[-N:]
    trace = search(data, starting_value=starting_value)
    elapsedTime = time.time() - startTime
    print(trace[-1], elapsedTime)

##################################################
#Non-linear optimization for SL-differentiation#
#################################################
def nonlinear_example():
    dataset = pd.read_excel("./output/SLdifferentiation.xlsx")
    target_group = 0.95
    T = target_group
    N = list(dataset['Item'])
    s = dict(zip(dataset.Period,dataset.s))# sigma prime
    Q = dict(zip(dataset.Period,dataset.Q)) # order quantity
    D = dict(zip(dataset.Period,dataset.D)) # yearly demand
    h = dict(zip(dataset.Period,dataset.h)) # unit price times holding cost percentage
    model = ConcreteModel(name="SL diff")
    model.x = Var(N, bounds=(0.85,1)) # z-value

    def obj_rule(model):
        return sum((4.85-(((Q[n]*(1-model.x[n]))/s[n])**1.3)*0.3924-(((Q[n]*(1-model.x[n]))/s[n])**0.135)*5.359)*h[n]*s[n] for n in N)
    model.obj = Objective(rule=obj_rule, sense=minimize)

    def group_service_level(model):
        return sum(D[n]*model.x[n] for n in N)/(sum(D[n] for n in N)) == T
    model.group_service_level = Constraint(rule=group_service_level)
    solver = SolverFactory('ipopt')
    solver.solve(model, tee=True)
    # print
    model.x.pprint()
    # or to dataframe with only the solution values per item.
    x = []
    for key in model.x:
        x.append(value(model.x[key]))
    dataset['optimal'] = pd.DataFrame(x)


##########################################
#Metropolis-Hastings algorithm#
##deviation voor lost sales omgevingen##
##########################################
def get_trace3(samples = 100, average=None, deviation=None, fill_rate=None, Q=None, order_level=None):
    average_current = average
    deviation_current = deviation
    order_level = order_level
    fill_rate = fill_rate
    Q = Q
    expected_bo_per_cycle = Q*(1-fill_rate)
    # Store the samples
    chain = numpy.array([0.]*samples)
    accept = np.inf
    proposal_width = 1.
    for i in range(samples):
        deviation_proposal = max(1,np.random.uniform(deviation_current-proposal_width ,deviation_current+proposal_width))
        current = (average-order_level)*(1-norm.cdf((order_level-average)/deviation_current, 0, 1))+deviation_current*norm.pdf((order_level-average)/deviation_current, 0, 1)
        proposal = (average-order_level)*(1-norm.cdf((order_level-average)/deviation_proposal, 0, 1))+deviation_proposal*norm.pdf((order_level-average)/deviation_proposal, 0, 1)
        accept_current = abs(current-expected_bo_per_cycle)
        accept_proposal = abs(proposal-expected_bo_per_cycle)
        if accept_proposal < accept_current:
            deviation_current = deviation_proposal
    return deviation_current

def metropolis_hasting_example4():
    average = 50
    deviation = 25
    fill_rate = 0.93
    order_level = 62
    Q = 100
    deviation = get_trace3(average=average, deviation=deviation, fill_rate=fill_rate, Q=Q, order_level=order_level)
    print(deviation)

##########################################
#Interview Question##
##########################################

def question():
    shops = [
    "BGI", "CDG", "DEL", "DOH", "DSM", "EWR", "EYW", "HND", "ICN",
    "JFK", "LGA", "LHR", "ORD", "SAN", "SFO", "SIN", "TLV", "BUD"
    ]

    routes = [
        ["DSM", "ORD"],
        ["ORD", "BGI"],
        ["BGI", "LGA"],
        ["SIN", "CDG"],
        ["CDG", "SIN"],
        ["CDG", "BUD"],
        ["DEL", "DOH"],
        ["DEL", "CDG"],
        ["TLV", "DEL"],
        ["EWR", "HND"],
        ["HND", "ICN"],
        ["HND", "JFK"],
        ["ICN", "JFK"],
        ["JFK", "LGA"],
        ["EYW", "LHR"],
        ["LHR", "SFO"],
        ["SFO", "SAN"],
        ["SFO", "DSM"],
        ["SAN", "EYW"]
    ]

    startingShop = "LGA"

    # Possible Solution (Steven P)

    temp = defaultdict(lambda: len(temp)) 
    res = [temp[ele] for ele in shops]
    zipbObj = zip(shops, res)
    dictOfWords = dict(zipbObj)
    for i in range(len(routes)):
        for j in range(len(routes[i])):
            routes[i][j] = dictOfWords.get(routes[i][j])
    V = len(shops)
    graph = Graph(V)
    for i in range(len(routes)):
        graph.add_edge(routes[i][0],routes[i][1])
    graph.get_scc()
    strong_components_dictionary = graph.dictionary_sc
    list_strong_components = list(strong_components_dictionary.keys())
    list_strong_components_members = list(strong_components_dictionary.values())
    list_of_members = [item for sublist in list_strong_components_members for item in sublist]
    # Now we got the strong components and their representatives and members.
    # We now have to make a new new adjency list. 
    # From this we can find the indegrees zero's.
    for i in range(len(routes)):
        if routes[i][0] in list_of_members:
            for j in range(len(list_strong_components_members)):
                if routes[i][0] in list_strong_components_members[j]:
                    x = j
                    routes[i][0] = list_strong_components[x]
                else: continue
        else:
            continue
    for i in range(len(routes)):
        if routes[i][1] in list_of_members:
            for j in range(len(list_strong_components_members)):
                if routes[i][1] in list_strong_components_members[j]:
                    x = j
                    routes[i][1] = list_strong_components[x]
                else: continue
        else:
            continue
    graph = Graph(V)
    for i in range(len(routes)):
        graph.add_edge(routes[i][0],routes[i][1])
    # Now loop over the dictionaries keys and remove all the values in the dict.values that are the same as the key. this leaves you the compressed direct graph.
    for key, value in graph.graph.items():
        while key in value:
            x = key
            value.remove(x)
    my_map = dict(graph.graph)
    l = list(my_map.values())
    flat_list = [item for sublist in l for item in sublist]
    # Now only add the "list_of_members" to the flat list and check which numbers are not in the list (5,6 and 16) or just count them!
    list_positive_indegrees = [] 
    [list_positive_indegrees.append(x) for x in (flat_list+list_of_members) if x not in list_positive_indegrees]
    list_positive_indegrees.sort()
    solution = len(shops)-len(list_positive_indegrees)
    list_positive_indegrees_shops = [k for k, v in dictOfWords.items() if v in set(list_positive_indegrees)]
    solution_shops = list(set(shops) - set(list_positive_indegrees_shops))
    print ("The shops that need an extra connection from shop LGA are : " + str(solution_shops)[1:-1])

##################################################
#smoothing constant via bayesian optimization##
#################################################

def mean_squared_error(observations, predictions):
    squared_error = (observations-predictions)**2
    return np.mean(squared_error)

def bayesian(data, samples = 5, starting_value=None):
    data = pd.DataFrame(data)
    def sampler(samples=None, data=None):
        errors = []
        for sample in samples:
            x = [starting_value]
            prediction = starting_value
            for j in range(len(data)-1):
                prediction = data.iloc[j, :]*sample + (1-sample)*prediction
                x.append(prediction.values[0])
            data['predictions'] = x
            error = mean_squared_error(data['Demand'].values, data['predictions'].values)
            errors.append(error)
            data = data[['Demand']]
        return errors, samples
    def surrogate(model, X):
    	# catch any warning generated when making a prediction
        with catch_warnings():
            # ignore generated warnings
            simplefilter("ignore")
            return model.predict(X)
    def opt_acquisition(X, y, model):
    	# random search, generate random samples
        Xsamples = np.array([round(np.random.rand(),4) for i in range(10)]).reshape(-1 ,1)
        # calculate the acquisition function for each sample
        scores = acquisition(X, Xsamples, model)
        # locate the index of the largest scores
        ix = np.argmin(scores)
        return Xsamples[ix, 0]
    # probability of improvement acquisition function
    def acquisition(X, Xsamples, model):
        # calculate the best surrogate score found so far
        yhat = surrogate(model, X)
        best = min(yhat)
        # calculate mean and stdev via surrogate function
        sample_output = surrogate(model, Xsamples)
        # calculate the probability of improvement
        scores = sample_output-best
        return scores
    samples = [round(np.random.rand(),4) for i in range(5)]
    errors, samples = sampler(samples=samples, data=data)
    model = MLPRegressor()
    model.fit(np.array(samples).reshape(-1, 1), np.array(errors))
    for i in range(50):
        # select the next point to sample
        x = opt_acquisition(np.array(samples).reshape(-1, 1), np.array(errors), model)
        actual_error, x = sampler(samples= [x], data=data)
        samples.extend(x)
        errors.extend(actual_error)
        model.fit(np.array(samples).reshape(-1, 1), np.array(errors))
    ix = np.argmin(np.array(errors))
    return np.array(samples).reshape(-1, 1)[ix], np.array(errors)[ix]

def bayesian_optimization():
    dataset = pd.read_excel("./output/Book1.xlsx")
    data = dataset['Demand']
    N = 100
    startTime = time.time()
    starting_value = np.mean(data[:-N])
    data = data[-N:]
    optimal_value, lowest_cost = bayesian(data, starting_value=starting_value)
    elapsedTime = time.time() - startTime
    print(optimal_value, elapsedTime)


def optimal_service_level():
    time_start = time.time()
    directory = get_config("directories", "service_levels")
    data = Optimizer.load('service_levels.xlsx', directory)
    data_sorted = data.sort_values('Demand', ascending =False)
    optimizer = Optimizer(constraint=0.96, data=data_sorted)
    optimizer.forward_pass()
    optimizer.backward_pass()
    data_sorted['Service_Level'] = data.index.map(optimizer.service_levels)
    data = data_sorted.sort_values('Item')
    lower_bound= get_config("bounds_service_levels", "lower_bound")
    upper_bound= get_config("bounds_service_levels", "upper_bound")
    data.Service_Level = data.apply(lambda row: max(lower_bound, min(upper_bound, row.Service_Level)), axis=1)
    elapsedTime = time.time() - time_start
    group_service_level = optimizer.group_service(data=data)
    optimizer.teunter(data=data)
    own_costs, teunter_costs = optimizer.calculate_costs()
    print(np.round(group_service_level, 2))
    print("The costs for the algorithm of Teunter et al. are: " + str(teunter_costs))
    print("The costs for our algorithm are: " + str(own_costs))
    print(-(1-own_costs/teunter_costs))
    print(optimizer.data)