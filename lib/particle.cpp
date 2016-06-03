#include <cstdlib>
#include <cassert>
#include "msg.h"
#include "util.h"
#include "fft.h"
#include "particle.h"

Particles::Particles(const int nc, const double boxsize_) :
  np_local(0)
{
  size_t nx= fft_local_nx(nc);
  
  size_t np_alloc= (size_t)((1.25*(nx + 1)*nc*nc));

  this->p= (Particle*) malloc(np_alloc*sizeof(Particle)); assert(p);
  this->force= (Float3*) calloc(3*np_alloc, sizeof(Float)); assert(force);
  this->np_allocated= np_alloc;
  this->boxsize= boxsize_;
  this->np_local= 0;
  this->np_total= 0;
  
  msg_printf(msg_verbose, "%lu Mbytes allocated for %lu particles\n",
	     mbytes(np_alloc*sizeof(Particle)), np_alloc);
}

Particles::~Particles()
{
  free(p);
}

