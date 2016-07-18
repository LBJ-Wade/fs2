import fs._fs as c
from fs.particles import Particles


"""COLA (COmoving Lagrangian Acceleration) is a numerical time integration
method using Lagrantial Perturbation Theory (LPT).
"""


def kick(particles, a_vel):
    """Update particle velocities to scale factor a_vel using COLA.

    Args:
        particles (Particles).
        a_vel (float): Scale factor after kick.
    """

    c._cola_kick(particles._particles, a_vel)


def drift(particles, a_pos):
    """Update particle positions to scale factor a_pos using COLA.

    Args:
        particles (Particles).
        a_pos (float): Scale factor after drift.
    """

    c._cola_drift(particles._particles, a_pos)
