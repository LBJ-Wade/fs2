#
# Test PM density grid
#
import numpy as np
import h5py
import fs
from create_pm_density import create_pm_density

fs.set_loglevel(0)

delta = create_pm_density()

#
# Test total density is number of particles
#

if fs.comm_this_node() == 0:
    nc = delta.shape
    nmesh = nc[0]*nc[1]*nc[2]
    eps = 1.0e-7

    a = delta.astype(np.float64)

    sum = np.sum(a)
    print("Total %e %e" % (sum, nmesh*eps))
    assert(abs(sum) < eps*nmesh)

    #
    # Compare with serial mesh
    #
    filename = 'pm_density.h5'
    file = h5py.File(filename, 'r')
    delta_ref = file['delta'][:]
    file.close()

    a = (delta - delta_ref).astype(np.float64)
    rms_error = np.std(a)
    max_error = np.max(np.abs(a))
    print('rms= %e' % rms_error)
    print('diff= %e' % max_error)
    assert(max_error < 1.0e-4)

fs.comm_mpi_finalise()
