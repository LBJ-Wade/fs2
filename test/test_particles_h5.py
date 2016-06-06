import h5py
import fs

omega_m = 0.308
nc = 4
pm_nc_factor = 1
boxsize = 4
a = 0.0
seed = 1


fs.cosmology_init(omega_m)
ps = fs.PowerSpectrum('../data/planck_matterpower.dat')

# Set 2LPT displacements at scale factor a
particles = fs.lpt(nc, boxsize, a, ps, seed)

fs.pm.init(nc*pm_nc_factor, pm_nc_factor, boxsize)

filename = 'particles_%d.h5' % fs.comm_n_nodes()

particles.save_hdf5(filename, 'ix')

np = nc*nc*nc


def assert_almost_equal(x, y):
    eps = 1.0e-15
    assert(abs(x - y) < eps)


if fs.comm_this_node() == 0:
    file = h5py.File(filename, 'r')
    file_nc = file['parameters/nc'][()]
    file_omegam = file['parameters/omega_m'][()]
    file_x = file['x'][:]
    file_id = file['id'][:]
    file.close()

    print(file_id.shape)
    assert(file_id.shape == (np,))
    assert(file_x.shape == (np, 3))

    print('data shape OK')

    #
    # Test parameters
    #
    assert(file_nc == nc)
    assert(file_omegam == omega_m)
    print('parameters OK')
    
    #
    # Test x_file
    #

    for i, a in enumerate(file_id):
        if a != i + 1:
            print('Error: particle id error in %s, %d != %d' %
                  (filename, a, i+1))
            raise AssertionError()

    print('%s/id OK' % filename)

    offset = 0.5
    for ix in range(nc):
        x = (ix + offset)*nc/boxsize
        for iy in range(nc):
            y = (iy + offset)*nc/boxsize
            for iz in range(nc):
                z = (iz + offset)*nc/boxsize
                index = (ix*nc + iy)*nc + iz
                assert_almost_equal(x, file_x[index, 0])
                assert_almost_equal(y, file_x[index, 1])
                assert_almost_equal(z, file_x[index, 2])

    print('%s/x  OK' % filename)

fs.comm_mpi_finalise()
