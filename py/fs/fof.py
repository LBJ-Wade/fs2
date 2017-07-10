import fs._fs as c
from fs.particles import Particles


def find_groups(particles, ll, *, quota=32, boxsize3=None, compute_nfof=False):
    """Run FoF halo finder find_groups(particles, ll, boxsize3=None, quota=32)

    Args:
        particles (Particles)
        ll (float): linking length

    Options:
        boxsize3 (Sequence of 3 floats): Length of the box enclosing particles
                  computed automatically if None (default)
        quota=32 (int): maximum number of particles in kdtree leaves

    Returns:
        nfof: an array of group sizes (number of FoF member particles)

    nfof[i] is the number of FoF members of group i (0 <= i < np_local)
    0 if particle i does not represent a group (i.e., belongs to a different group != i)
    """

    return c._fof_find_groups(particles._particles, ll, boxsize3, quota,
                              compute_nfof)
