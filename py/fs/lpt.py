import fs._fs as c
from fs.particles import Particles


def init(nc, boxsize, a, ps, seed, kind):
    """Generate 2LPT displacements and particle positions.

    This function generates a random Gaussian initial condition and
    create a grid of particles with 2LPT displacements. The velocities
    are 0.

    Args:
        nc (int): Number of particles per dimension;
                  number of particles np = nc**3.
        boxsize (float): length of the periodic box on a side [1/h Mpc].
        a (float): scale factor at which the positions are computed.
        ps (PowerSpectrum): Linear power spectrum extrapolated to a=1.
        seed (int): random seed for the random Gaussian initial density field
        kind (str): kind of xv, 'zeldovich', '2lpt', or 'cola'
                    cola sets v=0

    Returns:
        An instance of class Particles.
    """

    return Particles(_particles=c._lpt(nc, boxsize, a, seed, ps._ps,
                                       kind.lower()))


def set_offset(offset):
    """Set offset with respect to grid points
    x = (ix + offset)*dx,
    where ix is an integer, dx = boxsize/nc.

    Args:
        offset (float): offset (0 <= offset < 1)

    """

    c._set_offset(offset)


def set_zeldovich_force(particles, a):
    """Set Zel'dovich (1LPT) force to particles.force

    Args:
        a (float): scale factor of the force

    """

    c._set_zeldovich_force(particles._particles, a)
