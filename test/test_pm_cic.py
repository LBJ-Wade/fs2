#
# Test PM CIC density assignment with one particle
#
import unittest
import numpy as np
import fs


def one_particle_test(x, y, z):
    dx = boxsize/nc
    particles = fs.Particles(nc, boxsize)

    a = None
    if fs.comm.this_node() == 0:
        print("-- Testing %.1f %.1f %.1f -- " % (x*dx, y*dx, z*dx))
        a = np.zeros((1, 3))
        a[0, :] = [x*dx, y*dx, z*dx]

    print(a)
    particles.append(a)

    print('particle one')

    fs.pm.send_positions(particles)
    fft = fs.pm.compute_density(particles)
    a = fft.asarray()

    # Test Total = 1
    if fs.comm.this_node() == 0:
        total = np.sum(a + 1.0)
        if abs(total - 1.0) < 1.0e-15:
            print('%.2f OK' % total)
        else:
            print('%.2f Error' % total)

        assert(abs(total - 1.0) < 1.0e-15)


nc = 4
boxsize = 64
fs.msg.set_loglevel('debug')

np_buf = 10
fs.pm.init(nc, 1, boxsize)

one_particle_test(0, 2, 2)
one_particle_test(1, 2, 2)
one_particle_test(1.5, 2, 2)
one_particle_test(3, 2, 2)
one_particle_test(3.5, 2, 2)
one_particle_test(4, 2, 2)
