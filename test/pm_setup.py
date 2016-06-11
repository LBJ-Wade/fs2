#
# Create test data 'pm_density.h5' for test_pm_density.py
#
import h5py
import fs
import pm_setup
import sys


def setup_particles():
    # parameters
    omega_m = 0.308
    nc = 64
    pm_nc_factor = 1
    boxsize = 64
    a = 1.0
    seed = 1

    # initial setup
    fs.set_loglevel(3)
    fs.cosmology_init(omega_m)
    ps = fs.PowerSpectrum('../data/planck_matterpower.dat')

    # Set 2LPT displacements at scale factor a
    particles = fs.lpt.lpt(nc, boxsize, a, ps, seed)

    fs.pm.init(nc*pm_nc_factor, pm_nc_factor, boxsize)

    return particles


def density():
    particles = setup_particles()
    fft = fs.pm.compute_density(particles)
    return fft.asarray()


def force():
    particles = setup_particles()
    fs.pm.compute_force(particles)
    return particles
