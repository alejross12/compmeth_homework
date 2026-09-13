# Write a program that calculates the time it takes for a ball to drop from a user specified height to reach the ground.
# Use argparse. Allow the user to choose different values of gravity, and any other features you think maybe are interesting.
# Due Tuesday 9/15

import argparse
import numpy as np

parser = argparse.ArgumentParser(description = "Calculates the time it takes for a ball to hit the ground when dropped/thrown from a given height and other optional variables.")
parser.add_argument('d',         help = "Distance (in meters) from the ground the ball is dropped/thrown", type = float)
parser.add_argument('--g',       help = "Value of gravity, in m/s^2; default = 9.81", type = float, default = 9.81)
parser.add_argument('--v0',      help = "Initial velocity of the ball, in m/s; default = 0", type = float, default = 0)
parser.add_argument('--theta',   help = "Angle the ball is thrown relative to horizontal, in degrees; default = 90 (i.e. straight up)", type=float, default = 90)
parser.add_argument('--verbose', help = "Creates a verbose output with more details based on inputs", action='store_true')
args = parser.parse_args()

def main():
    # Calls arguments into variables to be used in the calculation
    d     = args.d
    g     = args.g
    v0    = args.v0
    theta = args.theta
    theta_rad = theta * (np.pi / 180)

    # Uses Newton's kinematic equation to solve for time, provided the above variables
    time = (v0 * np.sin(theta_rad) / g) + (1 / g) * np.sqrt((v0 * np.sin(theta_rad))**2 + 2*g*d)

    # An if/else tree to create a verbose output
    # Depending on what info the user provided, the wording of the output changes!
    if args.verbose:
        if v0 != 0:
            v0_verb = f" hit the ground after being thrown at {v0:.2f} m/s"
            h_verb  = f" from a height of {d:.2f} meters"

            if theta != 90:
                th_verb = f" {theta:.2f} degrees from horizontal and"
            else:
                th_verb = f" straight up"
        else:
            v0_verb = ''
            th_verb = ''
            h_verb  = f" drop {d:.2f} meters"

        if g != 9.81:
            g_verb = f" in {g:.2f} m/s^2 of gravity"
        else:
            g_verb = ''

        # Final output!
        output = f"It takes a ball {time:.2f} seconds to{v0_verb}{th_verb}{h_verb}{g_verb}."

    else:
        # Shortened output, for non-verbose
        output = f"{time:.2f} seconds"

    print(output)


if __name__ == '__main__':
    main()