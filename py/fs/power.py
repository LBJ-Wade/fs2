"""PowerSpectrum

Example:
    ps = fs.PowerSpectrum('data/planck_matterpower.dat')
    len(ps) # => number of data
    ps[i]   # => ith pair of (k, P)
"""

import fs._fs as c


class PowerSpectrum:
    """ps = PowerSpectrum(filename)
    Read tabulated text file, k P(k), compatible with CAMB matterpower.dat.

    Args:
        filename (str): Filename of tabulated power spectrum k P(k).

    Raises:
        IOError: If unable to read the file.
    """

    def __init__(self, filename):
        self._ps = c._power_alloc(filename)

    def __len__(self):
        """
        len(ps)
        Return number of k P pairs.
        """
        return c._power_n(self._ps)

    def __getitem__(self, i):
        """
        ps[i]

        Args:
            i (int): index of the data

        Returns:
            ith pair as a tuple (k, P).

        Raises:
            IndexError: If i is out of range
        """
        return c._power_i(self._ps, i)
