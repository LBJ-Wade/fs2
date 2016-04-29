from distutils.core import setup, Extension
import numpy as np

print("np.get_include()")
print(np.get_include())

setup(name='fs',
      version='0.0.1',
      author='Jun Koda',
      py_modules=[
      ],
      ext_modules=[
          Extension('fs._fs',
                    ['py_package.cpp', 'py_msg.cpp', # 'py_comm.cpp',
                    ],
                    include_dirs = ['../lib', np.get_include()],
                    library_dirs =  ['../lib'],
                    libraries = ['fs'],
          )
      ],
      packages=['fs'],
)


