#ifndef UTIL_H
#define UTIL_H 1

#include <string.h>
#include <assert.h>
#include "particle.h"

static inline size_t mbytes(size_t bytes)
{
  return bytes/(1024*1024);
}


static inline void periodic_wrapup_p(Particle* const p, const float_t boxsize)
{
  for(int k=0; k<3; k++) {
    if(p->x[k] < 0) p->x[k] += boxsize;
    if(p->x[k] >= boxsize) p->x[k] -= boxsize;

#ifdef CHECK
    assert(0 <= p->x[k] && p->x[k] < boxsize);
#endif
  }
}

#endif
