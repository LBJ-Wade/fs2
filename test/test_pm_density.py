#
# Test PM density grid
#
import numpy as np
import h5py
import fs
import pm_setup

fs.msg.set_loglevel(0)

delta = pm_setup.density()

#
# Test total density is number of particles
#

if fs.comm.this_node() == 0:
    nc = delta.shape
    nmesh = nc[0]*nc[1]*nc[2]
    eps = np.finfo(delta.dtype).eps

    a = delta.astype(np.float64)

    sum = np.sum(a)
    print("Total %e %e" % (sum, nmesh*eps))
    assert(abs(sum) < eps*nmesh)

    #
    # Compare with serial mesh
    #
    filename = 'pm_density_%s.h5' % fs.config_precision()
    file = h5py.File(filename, 'r')
    delta_ref = file['delta'][:]
    file.close()

    a = (delta - delta_ref).astype(np.float64)
    rms_error = np.std(a)
    max_error = np.max(np.abs(a))
    print('rms= %e' % rms_error)
    print('diff= %e' % max_error)
    assert(max_error < 500.0*eps)

    print('pm_density OK')
