import h5py
import fs
import pm_setup
import sys

delta = pm_setup.density()
filename = 'pm_density_%s.h5' % fs.config_precision()

file = h5py.File(filename, 'w')
file['delta'] = delta
file.close()
