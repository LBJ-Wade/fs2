import fs._fs as c
from fs.particles import Particles

def lpt(nc, boxsize, a, ps, seed):
    """Generate 2LPT positions.
    
    This function generates a random Gaussian initial condition and
    create a grid of particles with 2LPT displacements. The velocities
    are 0.

    Args:
        nc (int): Number of particles per dimension;
                  number of particles np = nc**3.
        boxsize (float): length of the periodic box on a side [1/h Mpc].
        a (float): scale factor at which the displacements are computed.
        ps (PowerSpectrum): Linear power spectrum extrapolated to a=1.
        seed (int): random seed for the random Gaussian initial density field

    """

    return Particles(_particles=c._lpt(nc, boxsize, a, seed, ps._ps))
