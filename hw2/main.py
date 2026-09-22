# Consider E(x) = integral 0 to x of e^(-t^2)
# a) Write program to calculate E(x) for values of x from 0 to 3, steps 0.1. Choose your method for integration and a suitable number of slices
# b) When convinced program is working, extend it further to make a graph of x vs. E(x)

import numpy as np
import matplotlib.pyplot as plt
import gaussxw as g
import argparse

parser = argparse.ArgumentParser(description = "Calculates and graphs the antiderivative of e^(-t^2) from 0 to x, for a range of discrete x values")
parser.add_argument('method', help = "Method used to calculate the integral at each datapoint: trap = Trapezoidal, simp = Simpson's, gauss = Gaussian", type = str, default = 10000, choices = ['trap', 'simp', 'gauss'])
parser.add_argument('--a',  help = "Lower bound of x; default = 0", type = float, default = 0)
parser.add_argument('--b',  help = "Upper bound of x; default = 3", type = float, default = 3)
parser.add_argument('--dx', help = "Step size of the x values b/w a and b; default = 0.1", type = float, default = 0.1)
parser.add_argument('--N',  help = "Number of slices used to calculate the integral at each x value; default = 10000, at most 20 for Gauss", type = int, default = 10000)

args = parser.parse_args()

def trap_integral(a, b, N, eq):
    '''
    Calculates the antiderivative of a function using the trapezoidal approximation method.

    Args:
        a  - lower bound of integral
        b  - upper bound of integral
        N  - number of slices
        eq - equation you are taking the integral of

    Returns:
        integral - value of the integral
    '''
    # Calculates step sizes b/w a and b given N
    dx  = (b - a) / N

    k   = np.array(range(N))
    sum = np.sum(eq(a + k*dx))

    integral = dx * (0.5*eq(a) + 0.5*eq(b) + sum)

    return integral

def simpsons_integral(a, b, N, eq):
    dx  = (b - a) / N

    k1   = np.array(range( int(N/2)) )
    sum1 = np.sum(eq(a + (2*k1 - 1) * dx))

    k2   = np.array(range( int(N/2 - 1) ))
    sum2 = np.sum(eq(a + 2 * k2 * dx))

    integral = (dx/3) * (eq(a) + eq(b) + 4 * sum1 + 2 * sum2)

    return integral


def gauss_integral(a, b, N, eq):
    x, w = g.gaussxw(N, a, b)
    integral = np.sum( w * eq(x) )

    return integral

def fxn(t):
    return np.exp(-t**2)

def main(method, a, b, dx, N):
    if method == 'trap':
        methfxn = trap_integral
        verbose = 'Trapezoidal'
    elif method == 'simp':
        methfxn = simpsons_integral
        verbose = "Simpson's"
    elif method == 'gauss':
        methfxn = gauss_integral
        verbose = "Gaussian"
        if N > 20:
            N = 20

    x = np.append(np.arange(a, b, dx), b)
    ploty = np.array([])
    for i in x:
        ploty = np.append(ploty, methfxn(0, i, N, fxn))

    plt.title(r'Graphing E(x) = $\int_{0}^{x}e^{-t^2}dt$ using ' + verbose + ' Method')
    plt.xlabel('x (Upper Bound of Integral)')
    plt.ylabel('E(x)')
    plt.plot(x, ploty, 'o', label = f'Step Size = {dx}\n# of slices = {N}')
    plt.legend(loc = 'lower right')
    plt.show()

if  __name__ == '__main__':
    method = args.method
    a      = args.a
    b      = args.b
    dx     = args.dx
    N      = args.N
    main(method, a, b, dx, N)