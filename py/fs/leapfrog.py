import fs._fs as c
from fs.particles import Particles


"""Leapfrog integration is a numerical method for time integration --
updating positions and velocities using velocities and forces, respectively.
"""


def set_initial_velocity(particles, a_vel):
    c._leapfrog_initial_velocity(particles._particles, a_vel)


def kick(particles, a_vel):
    """Update particle velocities to scale factor a_vel

    Args:
        particles (Particles)
        a_vel (float): Scale factor after kick.
    """

    c._leapfrog_kick(particles._particles, a_vel)


def drift(particles, a_pos):
    """Update particle positions to scale factor a_pos

    Args:
        particles (Particles)
        a_pos (float): Scale factor after drift.
    """

    c._leapfrog_drift(particles._particles, a_pos)
