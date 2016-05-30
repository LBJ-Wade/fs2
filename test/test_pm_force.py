#
# Compare PM force at particle locations
#
# mpirun -n 4 test_pm_force.py

import fs
import numpy as np
import h5py
import pm_setup

# read reference file
file = h5py.File('force_%s.h5' % fs.config_precision(), 'r')

ref_id = file['id'][:]
ref_force = file['f'][:]

file.close()

#print(ref_id)
#print(ref_force)

fs.set_loglevel(0)

particles = pm_setup.force()

particle_id = particles.id
particle_force = particles.force

assert(np.all(particle_id == ref_id))
print('pm_force id OK')

force_rms = np.std(ref_force)

diff = particle_force - ref_force

diff_rms = np.std(diff)
print('pm_force rms error %e / %e' % (diff_rms, force_rms))

diff_max = np.max(np.abs(diff))
print('pm_force max error %e / %e' % (diff_max, force_rms))

assert(diff_max < 1.0e-15)



#diff_id = np.abs(particle_id - ref_id)
#diff_id_max = np.max(diff_id)
#print(ref_id)
#print(particle_id)

#print(ref_force)
#print(particle_force)
# make data frame
# load id and force
# sort data frame with id
#delta_ref = file['delta'][:]
#file= h5py.file(

##if 'force.h5 
##particles.save_hdf5('force.h5', 'ixvf')
#use_long_id = False
#particles.save_gadget_binary('2lpt', use_long_id)
