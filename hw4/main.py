from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser(description = "Performs DFT on a given star's lightcurve, plotting its power spectrum and a comparison between the original and the post-DFT lightcurve.")
parser.add_argument('--file'     , help = 'File location of lightcurve fits file, default = ./tic0067719432.fits', type=str, default = "./tic0067719432.fits")
parser.add_argument('--l_jd'     , help = 'Lower boundary of the BJD date of the lightcurve; default = 1600', type=int, default=1600)
parser.add_argument('--u_jd'     , help = 'Upper boundary of the BJD date of the lightcurve; default = 2186', type=int, default=2186)
parser.add_argument('--lim_guess', help = 'Number of top-contributing fourier constants you want to keep for your inverse DFT OR starting value for the "scan" (see below); default = 100', type=int, default=100)
parser.add_argument('--scan'     , help = 'If flagged, performs DFT starting at the above "lim_guess", subtracting 5 until just before the inverse DFT no longer captures the behavior of the original lightcurve well; see code comments for more details.', action='store_true')
parser.add_argument('--errormax' , help = 'How big the highest difference between the original and inverse DFT should be before scan stops; default = 0.3, WARNING: increasing this value may lead to 0 contributions!', type=float, default=0.3)

args = parser.parse_args()

def readfile(file, l_jd, u_jd):
    '''
    Reads the lightcurve fits file and returns times and fluxes to be used for further analysis

    Args:
        file (str) - File path to fits file of lightcurve
        wl   (lst) - Wavelength range (lowerbound, upperbound) you want to work with, in nm

    Returns:
        times  (np.darray) - Array of times of the read lightcurve
        fluxes (np.darray) - Array of fluxes of the read lightcurve
    '''
    hdul = fits.open(file)

    times_raw  = hdul[1].data['times']
    fluxes_raw = hdul[1].data['fluxes']

    wl_range = (times_raw > l_jd) & (times_raw < u_jd)
    times  = times_raw[wl_range]
    fluxes = fluxes_raw[wl_range]

    return times, fluxes

def dft(y):
    '''
    Performs a (complex) Discrete Fourier Transform on a list of values serving as a function

    Args:
        y (np.darray) - List of values that you will perform DFT on

    Returns:
        c (np.darray) - Associated fourier coefficients for the function
    '''
    N = len(y)
    c = np.zeros(N, complex)
    xn = np.arange(0, N)
    for k in range(N):
        c[k] = np.sum(y * np.exp((-2j * np.pi * k * xn) / N))

    return c

def inv_dft(ck):
    '''
    Performs an (complex) Inverse Discrete Fourier Transform on a list of fourier coefficients

    Args:
        c (np.darray) - Associated fourier coefficients for the function

    Returns:
        y (np.darray) - List of values that serve as an "updated" lightcurve function
    '''
    N   = len(ck)
    y   = np.zeros(N, complex)
    k   = np.arange(0, N)
    for n in range(N):
        y[n] = (1/N) * np.sum(ck * np.exp((2j * np.pi * k * n) / N))

    return y

def idft_trim(ck, limit):
    '''
    Finds top contributing fourier coefficients and creates new fourier coefficients list with just
    those top contributors.

    Args:
        ck (np.darray) - Associated fourier coefficients for the function
        lim      (int) - How many of the top fourier coefficients are we keeping

    Returns:
        cknew (np.darray) - Updated fourier coefficients list with top contributors
    '''
    # Removes k = 0 frequency and takes absolute value to show all relative contributions
    ckabs = np.abs(ck[1:])

    # Sorts ck by max -> min contributions up to given limit of number of contributions
    # Returns indices, NOT values so next step is easier
    # + 1 because we removed k = 0, so all k numbers here are off by one
    maxk  = np.argsort(ckabs, descending=True)[:limit] + 1

    # Creates new array where ck contributions we keep have a 1 at their index and a 0 everywhere else
    ckidx = np.array([])
    for k in range(len(ck)):
        if k in maxk:
            val = 1
        else:
            val = 0
        ckidx = np.append(ckidx, val)

    # Combines the original ck with ckidx!
    # Creates new array where the ck contributions we want are at their respective index (and k),
    # but the contributions we don't want have been replaced with 0 (no more contribution at that k)
    cknew = ck * ckidx

    return cknew

def main(file, l_jd, u_jd, lim_guess, errormax, scan):
    # Reads file and sets variables to be used later
    times, fluxes = readfile(file, l_jd, u_jd)
    error = 0
    lim = lim_guess

    # DFT of fluxes to get ck array
    ck = dft(fluxes)

    # Changes behavior whether "--scan" was used
    if scan:
        ### IF --scan WAS USED
        # Performs idft on the top ck contributors given by variable "lim_guess" (default = 100)
        # If max difference between original fluxes and idft fluxes is < "errormax" (default = 0.25), then
        # we assume we don't need all of these contributions.
        # Removes bottom 5 contributions, finds new idft fluxes; repeats until max diff is > "errormax"
        # What results is the fewest ck contributions one can use before there is a significant difference
        # between the original fluxes and the post-dft fluxes!

        print(f"Scanning for fewest contributions needed, starting with the top {lim}...")
        while error < errormax:
            cknew = idft_trim(ck, lim)
            fluxes_idft = inv_dft(cknew)
            fluxes_new = fluxes_idft + (fluxes[0] - fluxes_idft[0])

            error = max(np.abs(fluxes - fluxes_new))
            if error < errormax:
                if lim == 0:
                    break
                lim -= 5

        print(f"Fewest contributions found at {lim}!")
    else:
        ### IF --scan WAS NOT USED
        # Performs idft on the top ck contributors given by variable "lim_guess" (default = 100)
        # Finds new fluxes post-dft!
        cknew = idft_trim(ck, lim)
        fluxes_idft = inv_dft(cknew)

        # Ensures that the post-dft fluxes start at the same point as the original fluxes
        fluxes_new = fluxes_idft + (fluxes[0] - fluxes_idft[0])

    fig1 = plt.figure(figsize=[15,10])
    plt.plot(np.abs(ck[1:]))
    fig1.suptitle("Power Spectrum of Fourier Coefficients", fontsize=26)
    plt.xlabel("Wavenumber (k)", fontsize=20)
    plt.xticks(fontsize = 16)
    plt.ylabel(r"Relative contribution $|c_k|$", fontsize=20)
    plt.yticks(fontsize = 16)
    plt.show()

    fig2 = plt.figure(figsize=[15,10])
    plt.plot(times, fluxes, label='Original Flux')
    plt.plot(times, np.abs(fluxes_new), label=f'Post-DFT Flux, {lim} coeff.')
    fig2.suptitle("Lightcurve of Object pre and post DFT", fontsize=26)
    plt.xlabel("Truncated Barycentric Julian Date", fontsize=20)
    plt.xticks(fontsize = 16)
    plt.ylabel("Normalized PDC Flux", fontsize=20)
    plt.yticks(fontsize = 16)
    plt.legend(loc = 'lower right', fontsize = 20)
    plt.show()

if  __name__ == '__main__':
    file      = args.file
    l_jd      = args.l_jd
    u_jd      = args.u_jd
    lim_guess = args.lim_guess
    errormax  = args.errormax
    scan      = args.scan
    main(file, l_jd, u_jd, lim_guess, errormax, scan)