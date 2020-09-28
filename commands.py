from config import get_config
import numpy as np
import pandas as pd
from scipy.stats import poisson
from concepts import bisection
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
        plt.xticks(np.arange(1, EOQ_no_waste, 5.0))
        plt.xticks(rotation=90)
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
    f = lambda x: ((o+p*x[0]+h*((x[0]*(x[0]-1))/5))/(1/(1-b)+x[0]))+(h*x[0])/(1+(x[0]*(1-b)))
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