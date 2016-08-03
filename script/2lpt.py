#
# Generate a 2LPT initial condition at a and save as gadget binary files
#
# mpirun -n 4 2lpt.py
#
import fs


# parameters
omega_m = 0.308
nc = 64
boxsize = 64
a = 0.1
seed = 1


# Initial setup
fs.set_loglevel(1)
fs.cosmology.init(omega_m)
ps = fs.PowerSpectrum('../data/planck_matterpower.dat')

# Set 2LPT displacements at scale factor a
particles = fs.lpt.init(nc, boxsize, a, ps, seed)

particles.save_gadget_binary('2lpt.gadget', False)


