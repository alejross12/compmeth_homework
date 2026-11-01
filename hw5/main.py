import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

parser = argparse.ArgumentParser(description='Performs a Monte Carlo simulation of a photon passing through a slab of constant electron density, and illustrating its simulated path')
parser.add_argument('-d', help='Distance photon will travel (in m) before simulation checks if it scatters; default = 50 m', type=float, default=50)
parser.add_argument('-w', help='Width of the slab (in m) the photon travels through before the simulation is completed; default = 1000 m', type=float, default=1000)
parser.add_argument('--animate', help='If flagged, the path of the photon will be animated; if not, you will see the whole path at once', action='store_true')

args = parser.parse_args()

def main(d, w, animate):

    ### Useful Variables:
    # Constants
    sigT = 6.652e-25    # Thompson cross-section (cm^2)
    ne   = 1e20         # Electron density of the slab (1/cm^3)
    d = d * 100         # Distance photon travels before scattering or not (m -> cm)
    w = w * 100         # Width of the slab photon is traveling through (m -> cm)
    p = ne * sigT * d   # Probability photon is scattered after distance d

    # Pre-loading variables
    rng = np.random.default_rng()

    distx = 0
    disty = 0
    theta = 0
    phi   = 0

    xlist     = np.array([])
    ylist     = np.array([])
    thetalist = np.array([])
    philist   = np.array([])

    ### While the photon is still in the slab, run the simulation:
    while distx < w:

        # If the photon scatters back to the start, restart the simulation
        if distx <= 0:
            print("Restarting!")
            # Resets all variables keeping track of the photon's path
            distx = 0
            disty = 0
            theta = 0
            phi   = 0

            xlist     = np.array([])
            ylist     = np.array([])
            thetalist = np.array([])
            philist   = np.array([])

        # Saves the current location of the photon to the appropriate list
        xlist     = np.append(xlist, distx)
        ylist     = np.append(ylist, disty)
        thetalist = np.append(thetalist, theta)
        philist   = np.append(philist, phi)

        # Based on the direction of the photon, calculates appropriate movement in 2D
        # Phi = [-pi/2, pi/2], from pointing directly into the screen to straight out
        # Theta = [0, 2pi], from pointing to the right and back around counterclockwise
        # Since we are only graphing in 2D, the phi's third dimension is reduced to its xy-plane projection
        r = d * np.cos(phi)
        distx += r * np.cos(theta)
        disty += r * np.sin(theta)

        # Randomly generates a value [0,1)
        # If value less than probability of scattering, photon is "scattered"
        # Assigns a random theta and phi based on above angles
        scatter = rng.random(1) < p
        if scatter:
            theta = 2 * np.pi * rng.random(1)
            phi   = np.pi * rng.random(1) - (np.pi / 2)

    # To ensure final position and angles are appended to appropriate lists
    xlist     = np.append(xlist, distx)
    ylist     = np.append(ylist, disty)
    thetalist = np.append(thetalist, theta)
    philist   = np.append(philist, phi)

    # Creates figure
    fig = plt.figure(figsize=[15,10])
    fig.suptitle(f'Photon Scattering in a {int(w/100)} m Slab of Constant Electron Density', fontsize=26)
    plt.xlabel('Horizontal Position of Photon (cm)', fontsize=20)
    plt.xticks(fontsize = 16)
    plt.ylabel('Vertical Position of Photon (cm)', fontsize=20)
    plt.yticks(fontsize = 16)
    plt.vlines(w, min(ylist), max(ylist), 'k')
    plt.ion()

    # If --animate is flagged, animates the path of the photon; otherwise, just plots the whole path
    if animate:
        # I know functions inside functions are generally frowned upon, but couldn't make it work otherwise (sorry)
        # Also not sure about that extra figure that shows up after you close the animation (also sorry)
        def update(frame):
            plt.plot(xlist[:frame], ylist[:frame], 'r')

        anim = animation.FuncAnimation(fig=fig, func=update, frames=len(xlist)+1, interval=10)
    else:
        plt.plot(xlist, ylist, 'r')

    plt.show(block=True)

if  __name__ == '__main__':
    d = args.d
    w = args.w
    animate = args.animate
    main(d, w, animate)