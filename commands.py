from config import get_config
from concepts.Automatic_differentiation.reverse_accumulation import reverse_autodiff, Overloader
from concepts.Gradient_boosting.regression_tree import prepare_dataset
import autograd.numpy as numpy
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
##########################################
def get_trace2(x, samples = 10000):
    beta = 0.0
    # Store the samples
    beta_chain = numpy.array([0.]*samples)
    beta_current = beta
    N = 100
    granularity = 100
    proposal_width = 0.07
    for i in range(samples):
        beta_proposal  = np.random.uniform(beta_current-proposal_width ,beta_current+proposal_width)
        # Compute the acceptance probability.
        # We will do both equations in log domain here to avoid underflow.
        accept = -log(granularity)+(N*log(beta_proposal)+sum(x-1)*log(1-beta_proposal))
        accept = accept-((-log(granularity))+(N*log(beta_current)+sum(x-1)*log(1-beta_current)))
        accept = min([1,accept])
        accept = exp(accept)
        if uniform.rvs(0,1) < accept:
            beta_current = beta_proposal
        beta_chain[i]=beta_current
    return beta_chain

def metropolis_hasting_example3():
    N=100
    data=np.random.geometric(0.35,N)
    x = data
    axsater_beta = 1-(2/(1+data.var()/data.mean()))
    trace = get_trace2(x)
    f, (ax1,ax2,ax3)=plt.subplots(3,1)
    # plot things
    ax2.plot(trace,'b')
    ax2.set_ylabel('$beta$')
    ax3.hist(trace, 50)
    ax3.set_xlabel('$beta$')

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
    X_train, y_train, X_test, y_test = prepare_dataset()
    pass

##########################################
#Reinforcement Learning#
##########################################