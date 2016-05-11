from distutils.core import setup, Extension
import numpy as np
import os

print("np.get_include()")
print(np.get_include())

os.environ["CC"] = "mpicc"
os.environ["CXX"] = "mpic++"

setup(name='fs',
      version='0.0.1',
      author='Jun Koda',
      py_modules=['fs.power', 'fs.particles', 'fs.functions'
      ],
      ext_modules=[
          Extension('fs._fs',
                    ['py_package.cpp', 'py_msg.cpp', 'py_comm.cpp',
                     'py_cosmology.cpp', 'py_power.cpp', 'py_particles.cpp',
                     'py_lpt.cpp', 'py_pm.cpp', 'py_cola.cpp','py_leapfrog.cpp',
                    ],
                    include_dirs = ['../lib', np.get_include()],
                    library_dirs =  ['../lib'],
                    libraries = ['fs'],
          )
      ],
      packages=['fs'],
)


# extra_compile_args=["-fopenmp"],
#                     extra_link_args=["-fopenmp"]
