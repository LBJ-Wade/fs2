#
# Test FFT
#

import unittest
import fs


class TestFFT(unittest.TestCase):
    def setUp(self):
        self.nc = 4
        fs.set_loglevel(0)
        self.fft = fs.FFT(self.nc)
        self.fft.set_test_data()

    def test_fft(self):
        nc = self.nc
        a = self.fft.asarray()
        if fs.comm_this_node() == 0:
            for ix in range(nc):
                for iy in range(nc):
                    for iz in range(nc):
                        index = (ix*nc + iy)*nc + iz
                        self.assertAlmostEqual(a[ix, iy, iz], index + 1)


if __name__ == '__main__':
    unittest.main()
