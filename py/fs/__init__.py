import atexit
from . import comm
from . import cosmology
from . import lpt
from . import pm
from . import cola
from . import leapfrog
from . import fof
from . import msg
from .particles import Particles
from .power import PowerSpectrum
from .fft import FFT
from .kdtree import KdTree

#from fs._fs import config_precision

_fs.comm_mpi_init()

atexit.register(_fs.comm_mpi_finalise)

