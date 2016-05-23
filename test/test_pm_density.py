#
# Test PM density grid
#
import unittest
import numpy as np
import h5py
import fs
from create_pm_density import create_pm_density

class TestFFT(unittest.TestCase):
    def setUp(self):
        filename = 'pm_density.h5'
        file = h5py.File(filename, 'r')
        
        self.delta_ref = file['delta']
        file.close()

        self.delta = create_pm_density()

    def tearDown(self):
        fs.comm_mpi_finalise()

    def test_total(self):
        delta = self.delta
        nc = delta.shape
        nmesh = nc[0]*nc[1]*nc[2]
        eps = 1.0e-15

        if fs.comm_this_node() == 0:
            a = delta.astype(np.float64)
        
            sum = np.sum(a)
            print(sum)
            self.assertTrue(sum < eps*nmesh)
            
    # ToDo: compare delta and delta_ref




if __name__ == '__main__':
    unittest.main()
