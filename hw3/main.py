from astropy import constants as c
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = 'Calculates the location of Lagrange points L1 and L2')
parser.add_argument('method', help = 'Method used to solve the Lagrange point equations', type = str, choices=['newtons','secant'])
parser.add_argument('point',  help = 'Lagrange point (L1 or L2) we are calculating', type = str, choices=['L1', 'L2'])

args = parser.parse_args()

# Useful constants that are used in calculations!
M_earth = c.M_earth.value
M_moon  = 7.348e22
G       = c.G.value
R       = 3.844e8
angvel  = 2.662e-6

def L1(r):
    # L1 equation given to us in homework, when all terms are moved to one side
    return ( (G * M_earth) / r**2 ) - ( (G * M_moon) / (R - r)**2 ) - angvel**2 * r

def dL1(r):
    # Derivative of the L1 equation
    return ( (-2 * G * M_earth) / r**3 ) - ( (2 * G * M_moon) / (R - r)**3 ) - angvel**2

def L2(r):
    # L2 equation; since L2 is on far side of Moon, acceleration components of Earth and Moon add instead of subtract
    return ( (G * M_earth) / r**2 ) + ( (G * M_moon) / (r - R)**2 ) - angvel**2 * r

def dL2(r):
    # Derivative of the L2 equation
    return ( (-2 * G * M_earth) / r**3 ) - ( (2 * G * M_moon) / (r - R)**3 ) - angvel**2

def newtons(eq, deq, r_guess, pverbose):
    '''
    Finds roots of equation using Newton's Method

    Args:
        eq       (fxn) - Equation we need to find roots of
        deq      (fxn) - Derivative of said equation
        r_guess  (flt) - Initial guess for solving root
        pverbose (str) - Lagrange point we are solving for

    Returns:
        root (int) - Root, or solution, to the given equation
    '''
    r = r_guess
    error = 999
    while error > 1e-3:
        # Finds value associated with guess
        guess  = eq(r)
        guessp = deq(r)

        # Calculates new "optimized" guess
        rp = r - guess / guessp

        # Checks if significant difference between original and "optimized" guess
        error = np.abs(r - rp)
        r = rp

    rint = int(r)
    print(f"Distance to {pverbose} is {rint:.3g} meters!")
    return rint

def secant(eq, r1_guess, r2_guess, pverbose):
    '''
    Finds roots of equation using Secant Method

    Args:
        eq       (fxn) - Equation we need to find roots of
        r1_guess (flt) - Initial guess for solving root, first point
        r2_guess (flt) - Initial guess for solving root, second point
        pverbose (str) - Lagrange point we are solving for

    Returns:
        root (int) - Root, or solution, to the given equation
    '''
    r1 = r1_guess
    r2 = r2_guess
    error = 999
    while error > 1e-3:
        # Finds value associated with guesses
        f_r1 = eq(r1)
        f_r2 = eq(r2)

        # Calculates new "optimized" guess
        guess       = f_r2
        guess_slope = (f_r2 - f_r1) / (r2 - r1)
        r3 = r2 - guess / guess_slope

        # Checks if significant difference between original and "optimized" guess
        error = np.abs(r2 - r3)
        r2 = r3

    rint = int(r2)
    print(f"Distance to {pverbose} is {rint:.3g} meters!")
    return rint

def main(method, point):

    # Calls variables associated with each Lagrange point
    if point == 'L1':
        # Associated functions
        fxn  = L1
        dfxn = dL1

        # Initial guesses for L1, using moon's distance as basis (has to be before it!)
        guess1 = R - 2e8
        guess2 = R - 100

        # Variables that help beautify things
        ylow  = -0.05
        yhigh = 0.01
        legendloc = 'lower left'
        pverbose = 'L1'
    elif point == 'L2':
        # Associated functions
        fxn  = L2
        dfxn = dL2

        # Initial guesses for L1, using moon's distance as basis (has to be before it!)
        guess1 = R + 1e8
        guess2 = R + 100

        # Variables that help beautify things
        ylow  = -0.01
        yhigh = 0.05
        legendloc = 'upper right'
        pverbose = 'L2'

    # Calculates solutions with method chosen
    if method == 'newtons':
        rint = newtons(fxn, dfxn, guess2, pverbose)
        mverbose = "Newton's"
    elif method == 'secant':
        rint = secant(fxn, guess1, guess2, pverbose)
        mverbose = "Secant"

    # x-values shown on graph
    if point == 'L1':
        xlow  = rint - 5e7
        xhigh = R + 5e7
    elif point == 'L2':
        xlow  = R - 5e7
        xhigh = rint + 5e7

    # Graphs Lagrange Equation
    rgraph = np.linspace(xlow, xhigh, 1000)
    plt.plot(rgraph, fxn(rgraph), label=f'{pverbose} Equation')

    # Horizontal line at y = 0
    plt.hlines(0, xlow, xhigh, 'tab:orange', linestyles='dashed')

    # Vertical line at Lagrange point
    plt.vlines(rint, ylow, yhigh, 'tab:red', label=f'Solution at {rint:.3g} m')

    # Vertical line at Moon's location
    plt.vlines(R, ylow, yhigh, 'tab:purple', label=f'Distance to Moon {R:.3g} m')

    # Beautifying the graph
    plt.title(f"Locating Lagrange Point using {mverbose} Method")
    plt.xlabel('Distance from Earth (r)')
    plt.ylabel('Value of Lagrange Equation (m/s^2)')
    plt.ylim(ylow, yhigh)
    plt.legend(loc = legendloc)
    plt.show()

if  __name__ == '__main__':
    method = args.method
    point  = args.point
    main(method, point)