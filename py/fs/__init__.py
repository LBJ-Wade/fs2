import atexit
import fs.comm
import fs.cosmology
import fs.lpt
import fs.pm
import fs.cola
import fs.leapfrog
import fs.fof
import fs.msg
from fs._fs import config_precision
from fs.particles import Particles
from fs.power import PowerSpectrum
from fs.fft import FFT
from fs.kdtree import KdTree


_fs.comm_mpi_init()

atexit.register(_fs.comm_mpi_finalise)

