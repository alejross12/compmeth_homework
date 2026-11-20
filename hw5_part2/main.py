import numpy as np
import time
import argparse

### Argparse ideas:
# N: running slab simulation N time (default 1000)
# sim_good: bad or good sun simulation?
# A: increase step size factor

parser = argparse.ArgumentParser(description = "Performs a Monte Carlo simulation to estimate how long it will take for a photon to travel through the Sun's atmosphere from its start at the center.")
parser.add_argument('-N',         help = 'Number of times Part 1 slab simulation will be run, default = 1000', type=int, default = 1000)
parser.add_argument('-A',         help = 'Step-increasing factor to make step sizes in Sun simulation larger (and hence simulation time), default = 10000\n' \
                                         'Estimated inaccurate sim times are: default (~15 sec), 5000 (~30 sec), ~1000 (~2.5 min)\n' \
                                         'Estimated accurate sim times are: 45-60 min)', type=int, default = 10000)
parser.add_argument('--sim_good', help = 'If flagged, will run the accurate but slow Sun simulation method', action = 'store_true')

args = parser.parse_args()

def slab_sim():
    '''
    Runs the exact simulation created in HW5 Part 1 (see that for more details)
    Only changes include:
        - No longer plots the path of the photon
        - slab length = 10l (or ten mean free paths)
        - Calculates the time the photon took to travel through the slab (see comments for details)

    returns:
        slab_time (float) - Time it took for the photon to travel through the slab,
                            in units of mean free travel time (l/c)
    '''
    rng  = np.random.default_rng()
    sigT  = 6.652e-25           # Thompson cross-section (cm^2)
    ne_slab = 1e20              # Electron density of the slab (1/cm^3)

    d  = 50 * 100               # Distance photon travels before scattering or not (50 m -> cm)
    p = ne_slab * sigT * d      # Probability of scattering
    l = 1 / (ne_slab * sigT)    # Mean free path (cm)
    w = 10 * l                  # Slab length, 10 mean free paths

    distx = 0
    disty = 0
    theta = 0
    phi   = 0

    its = 0
    while distx < w:
        if distx <= 0:
            distx = 0
            disty = 0
            theta = 0
            phi   = 0
            its   = 0

        r      = d * np.cos(phi)
        distx += r * np.cos(theta)
        disty += r * np.sin(theta)

        scatter = rng.random(1) < p
        if scatter:
            theta = 2 * np.pi * rng.random(1)
            phi   = np.pi * rng.random(1) - (np.pi / 2)

        its += 1

    slab_time = its * (d/100)/l

    return slab_time

def sun_sim_bad(timelist, A, N):
    '''
    Runs a Monte Carlo simulation of a photon traveling through the Sun (inaccurate but quick method)!

    Args:
        timelist (np.darray) - Posterior distribution of the time for a particle to travel through 10l slab
        A              (int) - Step-increasing factor to make step sizes larger (and hence simulation time)
        N              (int) - Number of times slab simulation was run

    Returns:
        timex        (float) - Estimated travel time of photon
    '''
    print('(You have selected the slightly inaccurate but quick method)')

    # Initiate variables
    rng  = np.random.default_rng()
    Rsun = 696340000   # Radius of Sun (m)
    w = Rsun * 0.9     # Slab length, 0.9 * radius of Sun (m)

    sigT  = 6.652e-25   # Thompson cross-section (cm^2)
    distx = 0
    timex = 0
    ne_x  = 0

    # Begin simulation
    while distx < w:
        # Each iteration, does the following:
        # - Calculates density at the current location of the photon
        # - Calculates mean free path (l) at current location
        # - Calculates step size, 10l times A (step-increasing factor)
        # - Adds step size to the current distance the photon traveled
        ne_x = ne(distx)        # (cm^-3)
        l = 1 / (ne_x * sigT)   # (cm)
        stepsize = 10 * l * A   # (cm)
        distx += stepsize/100   # (cm -> m)

        # - From the posterior distribution of times to travel through a slab of 10l,
        #   choose a random value to serve as the time it took to travel 10l and multiple by A,
        #   since photon traveled 10l A times, so it will take A times longer
        # - Random posterior value chosen by generating a random number b/w 0 and N, and calling
        #   the value at that random index
        idx = int(rng.random(1)[0] * N)
        timex += timelist[idx] * A

        # print('\n\n')
        # print(f'{int(distx)}/{int(w)} meters')
        # print(f'{timex} seconds')

    return timex

def sun_sim_good(timelist, A, N):
    '''
    Runs a Monte Carlo simulation of a photon traveling through the Sun (accurate but slow method)!

    Args:
        timelist (np.darray) - Posterior distribution of the time for a particle to travel through 10l slab
        A              (int) - Step-increasing factor to make step sizes larger (and hence simulation time)
        N              (int) - Number of times slab simulation was run

    Returns:
        timex        (float) - Estimated travel time of photon
    '''
    print('(You have selected the more accurate but slow method, estimated duration: 45-60 minutes)')

    # Initiate variables
    Rsun = 696340000   # Radius of Sun (m)
    w = Rsun * 0.9     # Slab length, 0.9 * radius of Sun (m)
    
    sigT  = 6.652e-25   # Thompson cross-section (cm^2)
    distx = 0
    timex = 0
    ne_x  = 0

    # Begin simulation
    while distx < w:
        # Each iteration, does the following:
        # - Calculates density at the current location of the photon
        # - Calculates mean free path (l) at current location
        # - Calculates step size, 10l times a step-increasing constant A
        # - Adds step size to the current distance the photon traveled
        ne_x = ne(distx)        # (cm^-3)
        l = 1 / (ne_x * sigT)   # (cm)
        stepsize = 10 * l * A   # (cm)
        distx += stepsize/100   # (cm -> m)

        # - From the posterior distribution of times to travel through a slab of 10l,
        #   choose a random value to serve as the time it took to travel 10l and multiple by A,
        #   since photon traveled 10l A times, so it will take A times longer
        # - Random posterior value chosen by generating a random number b/w 0 and N, and calling
        #   the value at that random index

        # - From the posterior distribution of times to travel through a slab of 10l,
        #   choose (A) random values to serve as the time it took to travel each of the A 10l steps,
        #   summing the times to get total time to travel 10l, A times.
        idx    = np.random.randint(0, N, A)
        timex += sum(timelist[idx])

    return timex

def ne(r):
    '''
    Electron density in the Sun's atmosphere as a function of distance from Sun's center

    Args:
        r  (float) - Distance from Sun's center, in meters

    Returns:
        ne (float) - Electron density at that location, in cm^-3
    '''
    ne = (2.5 * 10**(26)) * np.exp(-r / (0.096 * 696340000))
    return ne

def main(N, A, sim_good):
    '''
    Runs the entire code, including slab simulation and Sun simulation.

    Args:
        N         (int) - Number of times slab simulation was run
        A         (int) - Step-increasing factor to make step sizes larger (and hence simulation time)
        sim_good (bool) - If True, runs the accurate but slow method; if False, runs the inaccurate but quick method

    Returns:
        None
    '''
    print(f'---Running simulation of slab from Part 1---')
    print(f'Beginning {N} slab simulations...')

    tstart_slab = time.time()
    timelist = np.zeros(N)
    for i in range(N):
        slab_time = slab_sim()
        timelist[i] = slab_time

    tend_slab = time.time()
    print(f'Slab simulations complete (in {tend_slab - tstart_slab:.2f} seconds)!')
    print(f'\n')


    print('---Running simulation of a photon in the Sun---')
    print('Beginning Sun simulation...')

    tstart_sun = time.time()
    if sim_good:
        timex = sun_sim_good(timelist, A, N)
    else:
        timex = sun_sim_bad(timelist, A, N)

    tend_sun = time.time()
    print(f'Simulation complete (in {tend_sun - tstart_sun:.3f} seconds)!')
    print(f'Approximate travel time for this photon = {int(timex / 31556952)} years')

if  __name__ == '__main__':
    main(args.N, args.A, args.sim_good)