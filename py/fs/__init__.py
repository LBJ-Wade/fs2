import atexit
import fs.comm
import fs.cosmology
import fs.lpt
import fs.pm
import fs.cola
import fs.leapfrog
import fs.stat
import fs.fof
import fs.msg
from fs._fs import config_precision, timer_save
from fs.particles import Particles
from fs.power import PowerSpectrum
from fs.fft import FFT


_fs.comm_mpi_init()

atexit.register(_fs.comm_mpi_finalise)
