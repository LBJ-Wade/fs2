import numpy as np
import h5py
import fs
import cola_setup


def diff(x):
    if x > 0.5*boxsize:
        return boxsize - x
    return x


# reference data
filename = 'cola_%s.h5' % fs.config_precision()
file = h5py.File(filename, 'r')
x_ref = file['x'][:]
file.close()

# Cola simulation
particles = cola_setup.particles()

x_par = particles.x

# get boxsize a
# parallelise diff and get diff= x_par - x_ref
# need to periodic wrapup the difference
# must compare with dx=boxsize/nc

if fs.comm_this_node() == 0:
    a = x_par - x_ref
    a_max = np.max(np.abs(a))
    a_rms = np.std(a)
    print(a_max)
    print(a_rms)

    eps = np.finfo(x_par.dtype).eps
    assert(a_max < 1000.0*eps)
    assert(a_rms < 100.0*eps)

    print('max error %e; OK' % (a_max))
    print('rms error %e; OK' % (a_rms))

fs.comm_mpi_finalise()
