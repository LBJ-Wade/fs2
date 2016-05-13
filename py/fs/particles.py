import fs._fs as c

class Particles:
    def __init__(self, _particles):
        self._particles= _particles

    def __getitem__(self, i):
        print(i)
        return c._particles_getitem(self._particles, i)

    def __len__(self):
        return c._particles_len(self._particles)
    
    def slice(self, frac):
        return c._particles_slice(self._particles, frac)

    def save_gadget_binary(self, filename, use_longid=False):
        c._write_gadget_binary(self._particles, filename, use_longid)

    

