import fs._fs as c
from fs.particles import Particles


def find_groups(particles, **kwargs):
    """Run FoF halo finder

    Args:
        particles (Particles)

    Options:
        l=0.2 (float): linking parameter
        quota=32 (int): maximum number of particles in kdtree leaves

    Returns:
        an array of group sizes (number of FoF member particles)
    """

    l = kwargs.get('l', 0.2)
    quota = kwargs.get('quota', 32)

    return c._fof_find_groupsinit(particles._particles, l, quota)
