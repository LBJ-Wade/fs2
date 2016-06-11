import fs._fs as c
from fs.particles import Particles
from fs.fft import FFT


def cosmology_init(omega_m):
    c._cosmology_init(omega_m)


def leapfrog_initial_velocity(particles, a_vel):
    c._leapfrog_initial_velocity(particles._particles, a_vel)


def leapfrog_kick(particles, a_vel):
    c._leapfrog_kick(particles._particles, a_vel)


def leapfrog_drift(particles, a_pos):
    c._leapfrog_drift(particles._particles, a_pos)
