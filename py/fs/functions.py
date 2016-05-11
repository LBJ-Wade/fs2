import fs._fs as c
from fs.particles import Particles

def lpt(nc, boxsize, a, ps, seed):
    return Particles(c._lpt(nc, boxsize, a, seed, ps._ps))

def pm_compute_force(particles):
    c._pm_compute_force(particles._particles)

def cola_kick(particles, a_vel):
    c._cola_kick(particles._particles, a_vel)

def cola_drift(particles, a_pos):
    c._cola_drift(particles._particles, a_pos)

