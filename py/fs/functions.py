import fs._fs as c
from fs.particles import Particles

def lpt(nc, boxsize, a, ps, seed):
    return Particles(c._lpt(nc, boxsize, a, seed, ps._ps))
