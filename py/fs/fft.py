import fs._fs as c


class FFT:
    def __init__(self, arg):
        if isinstance(arg, int):
            self._fft = c._fft_alloc(arg)
        else:
            self._fft = arg

    def set_test_data(self):
        c._fft_set_test_data(self._fft)

    def asarray(self):
        return c._fft_fx_global_as_array(self._fft)
