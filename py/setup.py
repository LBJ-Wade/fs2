from distutils.core import setup, Extension
import numpy as np
import os

#
# $ make
#
# or set directories
# IDIRS: directories for headders -I
# LDIRS: directories for libraries -L
# DIRS:  libraries -lgsl ...

# directories for include -I(idir)

idirs = os.environ["IDIRS"]
if idirs:
    idirs = idirs.split()
else:
    idirs = []

idirs = ['../lib', np.get_include()] + idirs

# directories for libraries -L(dir)
ldirs = os.environ["LDIRS"]

if ldirs:
    ldirs = ldirs.split()
else:
    ldirs = []

    
# external libraries
libs = os.environ['LIBS'].split()
print('libs', libs)

lib_files = ['../lib/comm.cpp', '../lib/msg.cpp', '../lib/config.cpp',
             '../lib/fft.cpp', '../lib/mem.cpp', '../lib/particle.cpp',
             '../lib/util.cpp', '../lib/power.cpp',
             '../lib/cosmology.cpp', '../lib/lpt.cpp', '../lib/pm.cpp',
             '../lib/cola.cpp', '../lib/leapfrog.cpp',
             '../lib/pm_domain.cpp',
             '../lib/gadget_file.cpp',
             '../lib/kdtree.cpp', '../lib/fof.cpp',
]

# '../lib/hdf5_write.cpp',

py_files = ['py_package.cpp', 'py_msg.cpp', 'py_comm.cpp',
            'py_mem.cpp',
            'py_cosmology.cpp', 'py_power.cpp', 'py_particles.cpp',
            'py_lpt.cpp', 'py_pm.cpp', 'py_cola.cpp','py_leapfrog.cpp',
            'py_write.cpp', 'py_fft.cpp', #'py_hdf5_io.cpp',
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
                    include_dirs = idirs,
                    extra_compile_args = [os.environ["OPT"].strip()],
                    library_dirs =  ldirs,
                    libraries = libs,
                    undef_macros = ['NDEBUG'],
          )
      ],
      packages=['fs'],
)


# extra_compile_args=["-fopenmp"],
#                     extra_link_args=["-fopenmp"]
