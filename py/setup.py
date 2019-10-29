from setuptools import setup, Extension
import distutils.sysconfig
import numpy as np
import os


# Remove the "-Wstrict-prototypes" compiler option, which isn't valid for C++.
# https://stackoverflow.com/questions/8106258
# Remove -DNDEBUG option, which deactivate assert()
cfg_vars = distutils.sysconfig.get_config_vars()
for key, value in cfg_vars.items():
    if type(value) == str:
        cfg_vars[key] = value.replace('-Wstrict-prototypes', '').replace('-g', \
'').replace('-DNDEBUG', '')

#
# Includes -I
#
idirs = os.environ["IDIRS"]
if idirs:
    idirs = idirs.split()
else:
    idirs = []

idirs = [np.get_include(), ] + idirs

#
# Libraries -L
#
ldirs = os.environ["LDIRS"]

if ldirs:
    ldirs = ldirs.split()
else:
    ldirs = []

# external libraries set in Makefile (m, gsl, fftw)
libs = os.environ['LIBS'].split()
print('libs', libs)

#
# C++ codes
#
lib_files = ['comm.cpp', 'msg.cpp', 'config.cpp',
             'fft.cpp', 'mem.cpp', 'particle.cpp',
             'util.cpp', 'power.cpp',
             'cosmology.cpp', 'lpt.cpp', 'pm.cpp',
             'cola.cpp', 'leapfrog.cpp',
             'pm_domain.cpp',
             'gadget_file.cpp', 'hdf5_write.cpp',
             'kdtree.cpp', 'fof.cpp',
]

#
# Python interface
#
py_files = ['py_package.cpp', 'py_msg.cpp', 'py_comm.cpp',
            'py_mem.cpp',
            'py_cosmology.cpp', 'py_power.cpp', 'py_particles.cpp',
            'py_lpt.cpp', 'py_pm.cpp', 'py_cola.cpp','py_leapfrog.cpp',
            'py_write.cpp', 'py_fft.cpp', 'py_hdf5_io.cpp',
            'py_config.cpp',
            'py_fof.cpp', 'py_array.cpp', 'py_kdtree.cpp']


setup(name='fs',
      version='0.0.1',
      author='Jun Koda',
      py_modules=['fs.msg', 'fs.power', 'fs.particles', 'fs.functions',
                  'fs.lpt', 'fs.pm', 'fs.fof', 'fs.kdtree'],
      ext_modules=[
          Extension('fs._fs',
                    lib_files + py_files,
                    #depends = ['buffer.h', 'mask.h', 'np_array.h',
                    #],
                    extra_compile_args = ['-std=c++11'],
                    include_dirs = idirs,
                    library_dirs =  ldirs,
                    libraries = libs,
          )
      ],
      packages=['fs'],
      url='https://github.com/junkoda/fs2',
      license = "GPL3",
)
