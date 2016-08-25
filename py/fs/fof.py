import fs._fs as c
from fs.particles import Particles


def find_groups(particles, ll, **kwargs):
    """Run FoF halo finder find_groups(particles, ll, boxsize3=None, quota=32)

    Args:
        particles (Particles)
        ll (float): linking length

    Options:
        boxsize3 (Tuble of 3 Doubles): Length of the box enclosing particles
        quota=32 (int): maximum number of particles in kdtree leaves

    Returns:
        an array of group sizes (number of FoF member particles)
    """

    quota = kwargs.get('quota', 32)
    boxsize3 = kwargs.get('boxsize3', None)

    return c._fof_find_groups(particles._particles, ll, boxsize3, quota)
