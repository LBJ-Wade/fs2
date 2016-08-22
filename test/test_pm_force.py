#
# Test PM force parallelisation:
# check force does not depend on number of MPI nodes

import fs
import numpy as np
import h5py
import pm_setup

# read reference file
# $ python3 create_force_h5.py to create
file = h5py.File('force_%s.h5' % fs.config_precision(), 'r')

ref_id = file['id'][:]
ref_force = file['f'][:]

file.close()

# compute PM force
fs.msg.set_loglevel(0)

particles = pm_setup.force()
particle_id = particles.id
particle_force = particles.force

# compare two forces
if fs.comm.this_node() == 0:
    assert(np.all(particle_id == ref_id))
    print('pm_force id OK')

    force_rms = np.std(ref_force)

    diff = particle_force - ref_force

    diff_rms = np.std(diff)
    print('pm_force rms error %e / %e' % (diff_rms, force_rms))

    diff_max = np.max(np.abs(diff))
    print('pm_force max error %e / %e' % (diff_max, force_rms))

    eps = np.finfo(particle_force.dtype).eps

    assert(diff_rms < 20*eps)
    assert(diff_max < 1000*eps)
    print('pm_force OK')
