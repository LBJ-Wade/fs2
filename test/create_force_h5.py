import fs
import pm_setup

# parameters
omega_m = 0.308
nc = 64
boxsize = 64
a = 1.0
seed = 1

# initial setup
fs.set_loglevel(1)

particles = pm_setup.force()

filename = 'force_%s.h5' % fs.config_precision()

particles.save_hdf5(filename, 'if')

print('%s created.' % filename)
