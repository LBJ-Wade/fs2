from distutils.core import setup, Extension
import numpy as np
import os

print("np.get_include()")
print(np.get_include())

#os.environ["CC"] = "mpicc"
#os.environ["CXX"] = "mpic++"

setup(name='fs',
      version='0.0.1',
      author='Jun Koda',
      py_modules=['fs.msg', 'fs.power', 'fs.particles', 'fs.functions',
                  'fs.lpt', 'fs.pm', 'fs.fof',
      ],
      ext_modules=[
          Extension('fs._fs',
                    ['py_package.cpp', 'py_msg.cpp', 'py_comm.cpp',
                     'py_mem.cpp',
                     'py_cosmology.cpp', 'py_power.cpp', 'py_particles.cpp',
                     'py_lpt.cpp', 'py_pm.cpp', 'py_cola.cpp','py_leapfrog.cpp',
                     'py_write.cpp', 'py_fft.cpp', 'py_hdf5_io.cpp',
                     'py_config.cpp', 'py_timer.cpp', 'py_stat.cpp',
                     'py_fof.cpp', 'py_array.cpp',
                    ],
                    include_dirs = ['../lib', np.get_include()],
                    #define_macros = [('DOUBLEPRECISION','1')],
                    extra_compile_args = [os.environ["OPT"]],
                    library_dirs =  ['../lib'],
                    libraries = ['fs'],
          )
      ],
      packages=['fs'],
)


# extra_compile_args=["-fopenmp"],
#                     extra_link_args=["-fopenmp"]
