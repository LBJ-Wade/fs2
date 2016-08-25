# !!! This is for serial only

import unittest
import numpy as np
import fs
import signal

signal.signal(signal.SIGINT, signal.SIG_DFL)  # enable cancel with ctrl-c


nc = 4
boxsize = 1.0
dx = boxsize/nc
ll = 0.2*boxsize/nc

n = 10


def set_line(x, dx, n):
    """Return an 3-column array with x + dx*i for i = 0..n-1
    Args:
        x  (float[3]): 3-dim vector of the starting point.
        dx (float[3]): speration vector
        n  (int)     : number of output points
    """

    a = np.zeros((n, 3))
    for i in range(n):
        a[i, 0] = x[0] + i*dx[0]
        a[i, 1] = x[1] + i*dx[1]
        a[i, 2] = x[2] + i*dx[2]

    return a


class TestFFT(unittest.TestCase):
    def setUp(self):
        fs.msg.set_loglevel(3)

    def test_line_sparse(self):
        """Test FoF with a line of particles separated larger than
        the linking length. Expected to find n groups.
        """

        a = set_line([0.5*dx, 0.5*dx, 0.5*dx], [1.1*ll, 0, 0], n)
        particles = fs.Particles(nc, boxsize)
        particles.append(a)

        nfof = fs.fof.find_groups(particles, ll, quota=16)
        self.assertEqual(len(nfof), n)

        nfof = fs.fof.find_groups(particles, ll, quota=2)
        self.assertEqual(len(nfof), n)

        return True

    def test_line_dense(self):
        """Test FoF with a line of particles separated smaller than
        the linking length. Expected to find 1 groups.
        """

        a = set_line([0.5*dx, 0.5*dx, 0.5*dx], [0.99*ll, 0, 0], n)
        particles = fs.Particles(nc, boxsize)
        particles.append(a)

        nfof = fs.fof.find_groups(particles, ll, quota=16)
        self.assertEqual(len(nfof), 1)

        nfof = fs.fof.find_groups(particles, ll, quota=2)
        self.assertEqual(len(nfof), 1)


if __name__ == '__main__':
    unittest.main()
