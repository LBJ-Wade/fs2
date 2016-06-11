import fs._fs as c
from fs.particles import Particles
from fs.fft import FFT


def init(nc_pm, pm_factor, boxsize):
    """Initialise pm module.

    Args:
        np_pm (int): Number of mesh per dimension
        pm_factor (int): nc_pm/nc -- number of mesh / particle per dimension
        boxsize (float): Length of the periodic box
    """

    c._pm_init(nc_pm, pm_factor, boxsize)


def compute_force(particles):
    """Compute particles.force from particles.x

    Prerequisite:
        - call pm.init.
        - set particles.x

    Args:
        particles (Particles)
    """

    c._pm_compute_force(particles._particles)


def compute_density(particles):
    """Compute density mesh from particles.x

    Prerequisite:
        - call pm.init
        - set particles.x

    Args:
        particles (Particles)

    Returns:
        fft (FFT): density mesh
    """

    _fft = c._pm_compute_density(particles._particles)
    return FFT(_fft)


def domain_init(particles):
    """Initialise pm_domain module.

    This function is called automatically. No need to call, usually.

    Args:
        particles (Particles)
    """

    c._pm_domain_init(particles._particles)


def send_positions(particles):
    """Send particle positions to relevant PM domains.

    This function is called automatically.

    Args:
        particles (Particles)
    """

    c._pm_send_positions(particles._particles)


def write_packet_info(filename):
    """Write MPI particle exchange information to a HDF5 file.

    Args:
        filename (str): output file name.
    """

    c._pm_write_packet_info(filename)
