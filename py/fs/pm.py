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


def force(particles):
    """Compute particles.force from particles.x

    Prerequisite:
        pm.init()

    Args:
        particles (Particles)
    """

    send_positions(particles)
    compute_density(particles)
    check_total_density()
    compute_force(particles)
    get_forces(particles)


def send_positions(particles):
    """Send particle positions to other MPI nodes

    Prerequisit:
        pm.init().
    """
    c._pm_send_positions(particles._particles)


def compute_density(particles):
    """Compute density mesh from particles.x.

    Prerequisite:
        pm.send_positions()
        particles.x

    Args:
        particles (Particles)

    Returns:
        delta (FFT): density mesh.
    """
    _fft = c._pm_compute_density(particles._particles)
    return FFT(_fft)


def check_total_density():
    """Check total density: <delta> = 0.

    Prerequisite:
        pm.compute_density().

    Raises:
        AssertionError: If the density is not zero
        within floating point precision.
    """
    c._pm_check_total_density()


def compute_force(particles):
    """Compute particles.force from density mesh

    Prerequisite:
        pm.compute_density().

    Args:
        particles (Particles).
    """
    c._pm_compute_force(particles._particles)


def get_forces(particles):
    """Get particle.force from other MPI nodes

    Prerequisite:
        - pm.send_particles(), pm.compute_force().

    Args:
        particles (Particles).
    """
    c._pm_get_forces(particles._particles)


def domain_init(particles):
    """Initialise pm_domain module.

    This function is called automatically. No need to call usually.

    Args:
        particles (Particles)
    """
    c._pm_domain_init(particles._particles)


def write_packet_info(filename):
    """Write MPI particle exchange information to a HDF5 file.

    Args:
        filename (str): output file name.
    """
    c._pm_write_packet_info(filename)


def set_packet_size(packet_size):
    """Set packet size for PM position exchange.

    To change the packet_size from the default value, this function must be
    called before any PM force computation.

    Args:
        packet_size (int): number of floats in the packet
    """
    c._pm_set_packet_size(packet_size)
