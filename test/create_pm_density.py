#
# Create test data 'pm_density.h5' for test_pm_density.py
#

import h5py
import fs

# parameters
omega_m = 0.308
h = 0.67
nc = 64
pm_nc_factor = 1
boxsize = 64
a = 1.0
seed = 1

# initial setup
fs.set_loglevel(1)
fs.cosmology_init(omega_m)
ps = fs.PowerSpectrum('../data/planck_matterpower.dat')
fs.pm_init(nc*pm_nc_factor, pm_nc_factor, boxsize)

# Set 2LPT displacements at scale factor a
particles = fs.lpt(nc, boxsize, a, ps, seed)


fft = fs.pm_compute_density(particles)
a = fft.asarray()

filename = 'pm_density.h5'
file = h5py.File(filename, 'w')

file.close()
file['delta'] = a

print("%s written\n" % filename)
