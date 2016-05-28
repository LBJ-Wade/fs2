import fs._fs as c


class Particles:
    def __init__(self, nc=0, boxsize=0.0, **kwargs):
        # Particles(nc, boxsize) or Particles(_particles=particles)
        if '_particles' in kwargs:
            self._particles = kwargs['_particles']
        else:
            self._particles = c._particles_alloc(nc, boxsize)

    def __getitem__(self, i):
        return c._particles_getitem(self._particles, i)

    def __len__(self):
        return c._particles_len(self._particles)

    def set_one(self, x, y, z):
        c._particles_one(self._particles, x, y, z)

    def update_np_total(self):
        c._particles_update_np_total(self._particles)
        
    def slice(self, frac):
        return c._particles_slice(self._particles, frac)

    def save_gadget_binary(self, filename, use_longid=False):
        c._write_gadget_binary(self._particles, filename, use_longid)

    def save_hdf5(self, filename, var):
        c._hdf5_write_particles(self._particles, filename, var)
