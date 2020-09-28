from config import get_config
import numpy as np
import pandas as pd
from scipy.stats import poisson
from concepts import bisection

import warnings
warnings.filterwarnings("ignore")

#####################
#    OPTIMAL ORDER QUANTITY WITH WASTE     #
#####################


# p = 10
# w = 10
# h = 0.25
# o = 15
# x_m = 10
# d = 1000
# f = lambda x: p*h*0.5-((d*o)/(x-(x*poisson.pmf(x_m,x,loc=0)+(x-x_m)*(1-poisson.cdf(x_m,x,loc=0))))**2)*(1-poisson.cdf(x,x_m,loc=0))+w*d*((-1*(x*poisson.pmf(x_m,x,loc=0)+(x-x_m)*(1-poisson.cdf(x_m,x,loc=0)))/x**2)+(poisson.cdf(x,x_m,loc=0)/x))
# EOQ_no_waste = np.sqrt((2*d*o)/(h*p))
# EOQ_with_waste = bisection(f,1,EOQ_no_waste,200)
# for Q in range(1, int(EOQ_no_waste)):
#     F_x = poisson.cdf(x_m,Q,loc=0)
#     EO = (Q*poisson.pmf(x_m,Q,loc=0)+(Q-x_m)*(1-F_x))
#     IC = h*p*Q/2
#     BC = (d*o)/(Q-EO)
#     WC = EO*w*(d/Q)
#     cost = IC+BC+WC
#     print(Q,cost)
# print(EOQ_with_waste)

def hooke_reeves_example():
    p = 10
    i = 0.25
    h = i*p
    o = 15
    d = 1000
    b = 0.25
    f = lambda x: x[0]**2 + x[0]**x[1]+3*x[1]**2
    # sample functions:
        # ((o+p*x[0]+h*((x[0]*(x[0]-1))/5))/(1/(1-b)+x[0]))+(h*x[0])/(1+(x[0]*(1-b)))
        # (d/x[0])*o+h*x[0]/2
        # x[0]**2 + x[0]**x[1]+3*x[1]**2
    solution = hooke_reeves(f, [2,3], 1, 0.01, [-10., -10.])
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