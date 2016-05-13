#
# Generate a 2LPT initial condition at a and save as gadget binary files
#
# mpirun -n <number of nodes> 2lpt.py

import fs

# parameters
omega_m = 0.308
nc = 64
boxsize = 64
a = 0.1
seed = 1

# initial setup
fs.set_loglevel(1)
fs.cosmology_init(omega_m)
ps = fs.PowerSpectrum('../data/planck_matterpower.dat')

# Set 2LPT displacements at scale factor a
particles = fs.lpt(nc, boxsize, a, ps, seed)

use_long_id= False
particles.save_gadget_binary('2lpt', use_long_id)

