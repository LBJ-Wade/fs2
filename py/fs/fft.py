import fs._fs as c


class FFT:
    """3-dimensional grid in real or Fourier space

    Args:
        nc (int): Allocate a new FFT grid with nc grids per dim
        _fft (_FFT): Wrap a C++ FFT pointer
    """

    def __init__(self, arg):
        if isinstance(arg, int):
            self._fft = c._fft_alloc(arg)
        else:
            self._fft = arg

    def set_test_data(self):
        """Set test data
        grid(ix, iy, iz) = (ix*nc + iy)*nc + iz
        """

        c._fft_set_test_data(self._fft)

    def asarray(self):
        """Return the grid as a 3-dimensional np.array

        Returns:
           np.array for node 0; shape is (nc, nc, nc)
           None for other nodes
        """

        return c._fft_fx_global_as_array(self._fft)
