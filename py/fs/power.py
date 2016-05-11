import fs._fs as c

class PowerSpectrum:
    def __init__(self, filename):
        self._ps = c._power_alloc(filename)

    def __len__(self):
        return c._power_n(self._ps)

    def __getitem__(self, index):
        # ToDo: handle slices, too
        return c._power_i(self._ps, index)
