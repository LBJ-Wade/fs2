import fs._fs as c


class Particles(object):
    """
    A set of particles.
    Use lpt() to create a set of particles.

    Attributes:
        np_total [unsigned long]: total number of particles
        id (np.array [uint64]): all particle IDs for node 0,
                                `None` for node > 0.
        x (np.array [float]):  all particle positions for node 0
        force (np.array [float]):  all particle velocities for node 0
    """
    def __init__(self, nc=0, boxsize=0.0, **kwargs):
        # Particles(nc, boxsize) or Particles(_particles=particles)
        if '_particles' in kwargs:
            self._particles = kwargs['_particles']
        else:
            self._particles = c._particles_alloc(nc, boxsize)

    def __getitem__(self, index):
        """
        a = particles.__getitem__(i:j:k) <==> particles[i:j:k]

        Returns:
            local particles as a np.array.

        * particles[i]:     ith particles.
        * particles[i:j]:   particles with indeces [i, j).
        * particles[i:j:k]: particles with indeces i + k*n smaller than j.


        Args:
           index: an integer or a slice i:j:k

        Returns:
            particle data in internal units as np.array.

            * a[0:3]:  positions.
            * a[3:6]:  velocities (internal unit).
            * a[6:9]:  1st-order LPT displacements.
            * a[9:12]: 2nd-order LPT displacements.
        """
        return c._particles_getitem(self._particles, i)

    def __len__(self):
        """
        len(particles)

        Returns:
            local number of particles in this MPI node.
        """
        return c._particles_len(self._particles)

    def set_one(self, x, y, z):
        """
        Set one particle at x y z.

        Args:
            float x y z: positions
        """
        c._particles_one(self._particles, x, y, z)

    def update_np_total(self):
        c._particles_update_np_total(self._particles)

    def slice(self, frac):
        return c._particles_slice(self._particles, frac)

    def save_gadget_binary(self, filename, use_longid=False):
        """
        Args:
            filename (str): Output file name.
            use_longid (bool): Write 8-byte ID (default: False)
        """
        c._write_gadget_binary(self._particles, filename, use_longid)

    def save_hdf5(self, filename, var):
        """
        Args:
            filename (str): Output file name.
            var (str): Output variables, a subset of 'ixvf12' in this order.

                * i: ID
                * x: positions
                * v: velocities
                * f: force
                * 1: 1LPT displacements
                * 2: 2nd-order displacements
        """
        c._hdf5_write_particles(self._particles, filename, var)

    @property
    def np_total(self):
        return c._particles_np_total(self._particles)

    @property
    def id(self):
        return c._particles_id_asarray(self._particles)

    @property
    def x(self):
        return c._particles_x_asarray(self._particles)

    @property
    def force(self):
        return c._particles_force_asarray(self._particles)
