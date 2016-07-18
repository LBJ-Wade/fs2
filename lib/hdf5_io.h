#ifndef HDF5_IO_H
#define HDF5_IO_H 1

#include "particle.h"

void hdf5_write_particles(const char filename[],
			  Particles const * const particles,
			  char const* var);

void hdf5_write_packet_data(const char filename[], const int data[], const int n);
#endif
