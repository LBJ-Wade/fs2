#
# Test PM CIC density assignment
#

import unittest
import numpy as np
import fs

nc = 4
boxsize = 64

class TestFFT(unittest.TestCase):
    def setUp(self):
        fs.set_loglevel(3)

        self.particles= fs.Particles(nc, boxsize)
        fs.pm_init(nc, 1, boxsize)
        
    def tearDown(self):
        fs.comm_mpi_finalise()

    def test_one(self):
        xs = [0.0, 0.25, 0.5, 1.0, nc/2, nc-0.25, nc]
        for x in xs:
            for y in xs:
                for z in xs:
                    self.one_particle_test(x, y, z)
    
    def one_particle_test(self, x, y, z):
        dx= boxsize/nc
        self.particles.set_one(x*dx, y*dx, z*dx)
        fft = fs.pm_compute_density(self.particles)
        a = fft.asarray() + 1.0

        # Test Total = 1
        total= np.sum(a)
        self.assertTrue(abs(total - 1.0) < 1.0e-15)

        # ToDo? Test array 'a'
        # Test tes


if __name__ == '__main__':
    unittest.main()
