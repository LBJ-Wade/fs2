#ifndef HDF5_IO_H
#define HDF5_IO_H 1

#include "particle.h"

void hdf5_write_particles(const char filename[],
			  Particles const * const particles,
			  char const* var);

#endif
