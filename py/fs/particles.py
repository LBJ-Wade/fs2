import fs._fs as c

class Particles:
    def __init__(self, _particles):
        self._particles= _particles

    def __getitem__(self, i):
        print(i)
        return c._particles_getitem(self._particles, i)
    
    def slice(self, frac):
        return c._particles_slice(self._particles, frac)

    

