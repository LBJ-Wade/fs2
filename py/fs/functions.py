import fs._fs as c
from fs.particles import Particles
from fs.fft import FFT


def cosmology_init(omega_m, h=1):
    c._cosmology_init(omega_m, h)


def pm_compute_force(particles):
    c._pm_compute_force(particles._particles)


def pm_compute_density(particles):
    _fft = c._pm_compute_density(particles._particles)
    return FFT(_fft)


def cola_kick(particles, a_vel):
    c._cola_kick(particles._particles, a_vel)


def cola_drift(particles, a_pos):
    c._cola_drift(particles._particles, a_pos)


def leapfrog_initial_velocity(particles, a_vel):
    c._leapfrog_initial_velocity(particles._particles, a_vel)


def leapfrog_kick(particles, a_vel):
    c._leapfrog_kick(particles._particles, a_vel)


def leapfrog_drift(particles, a_pos):
    c._leapfrog_drift(particles._particles, a_pos)
