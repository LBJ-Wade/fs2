#include <cstdlib>
#include <cassert>
#include "msg.h"
#include "util.h"
#include "fft.h"
#include "particle.h"

Particles::Particles(const int nc, const double boxsize_)
{
  size_t nx= fft_local_nx(nc);
  
  size_t np_alloc= (size_t)((1.25*(nx + 1)*nc*nc));

  p= (Particle*) malloc(np_alloc*sizeof(Particle)); assert(p);
  force= (float3*) calloc(3*np_alloc, sizeof(float)); assert(force);
  np_allocated= np_alloc;
  boxsize= boxsize_;

  
  msg_printf(msg_verbose, "%lu Mbytes allocated for %lu particles\n",
	     mbytes(np_alloc*sizeof(Particle)), np_alloc);
}

Particles::~Particles()
{
  free(p);
}

