#
# FoF halo finder
#
# python3 fof.py
#
import signal
import fs


signal.signal(signal.SIGINT, signal.SIG_DFL) # enable cancel with ctrl-c

# parameters
omega_m = 0.308
nc = 64
boxsize = 64
a = 1
seed = 1


# Initial setup
fs.msg.set_loglevel('verbose')
fs.cosmology.init(omega_m)
ps = fs.PowerSpectrum('../data/planck_matterpower.dat')

# Set 2LPT displacements at scale factor a
particles = fs.lpt.init(nc, boxsize, a, ps, seed)

a = fs.fof.find_groups(particles)
print(a)

