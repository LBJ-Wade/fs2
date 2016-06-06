#ifndef GADGET_FILE_H
#define GADGET_FILE_H 1

#include "particle.h"

typedef struct {
  int      np[6];
  double   mass[6];
  double   time;
  double   redshift;
  int      flag_sfr;
  int      flag_feedback;
  unsigned int np_total[6];
  int      flag_cooling;
  int      num_files;
  double   boxsize;
  double   omega0;
  double   omega_lambda;
  double   hubble_param; 
  int flag_stellarage;
  int flag_metals;
  unsigned int np_total_highword[6];
  int  flag_entropy_instead_u;
  char fill[60];
} GadgetHeader;

void gadget_file_write_particles(const char filebase[],
				 Particles const * const particles,
				 const int use_long_id, const double h=1.0);

#endif
