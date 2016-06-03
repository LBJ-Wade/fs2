import fs
import cola_setup


particles = cola_setup.particles()

filename = 'cola_%s.h5' % fs.config_precision()

particles.save_hdf5(filename, 'xv')
print('%s written' % filename)

fs.comm_mpi_finalise()
