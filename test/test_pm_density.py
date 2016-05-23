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
    eps = 1.0e-15

    a = delta.astype(np.float64)

    sum = np.sum(a)
    print("Total %.3f" % sum)
    assert(abs(sum) < eps*nmesh)

#
# Compare with serial mesh
#
    filename = 'pm_density.h5'
    file = h5py.File(filename, 'r')
    delta_ref = file['delta'][:]
    file.close()

    a = (delta - delta_ref).astype(np.float64)
    diff = np.max(np.abs(a))
    print('diff= %e' % diff)
    assert(diff < 1.0e-7)

fs.comm_mpi_finalise()
