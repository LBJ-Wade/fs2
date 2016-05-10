//
// Definition of particle & particles data structure
//

#ifndef PARTICLE_H
#define PARTICLE_H 1

#include <stdint.h>
#include "config.h"

struct Particle {
  float_t  x[3];
  float_t  v[3];
  float_t  dx1[3];  // 1LPT (ZA) displacement
  float_t  dx2[3];  // 2LPT displacement
  uint64_t id;      // Particle index 1,2,3...
};

struct Particles {
  Particle* p;
  double a_x, a_v;
  float3* force;

  size_t np_local, np_allocated;
  uint64_t np_total;
  double omega_m, boxsize;
};

#endif
