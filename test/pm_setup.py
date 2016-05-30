#
# Create test data 'pm_density.h5' for test_pm_density.py
#
import h5py
import fs
import sys
    
def setup_particles():
    # parameters
    omega_m = 0.308
    h = 0.67
    nc = 64
    pm_nc_factor = 1
    boxsize = 64
    a = 1.0
    seed = 1

    # initial setup
    #fs.set_loglevel(1)
    fs.cosmology_init(omega_m)
    ps = fs.PowerSpectrum('../data/planck_matterpower.dat')


    # Set 2LPT displacements at scale factor a
    particles = fs.lpt(nc, boxsize, a, ps, seed)

    fs.pm_init(nc*pm_nc_factor, pm_nc_factor, boxsize)

    return particles

def density():
    particles = setup_particles()
    fft = fs.pm_compute_density(particles)
    return fft.asarray()

def force():
    particles = setup_particles()
    fs.pm_compute_force(particles)
    return particles

if __name__ == '__main__':
    
    if sys.argv[-1] == 'density':
        delta = pm_density()
        file = h5py.File('pm_density.h5', 'w')
        file['delta'] = delta
        file.close()

