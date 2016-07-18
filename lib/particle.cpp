#include <cstdlib>
#include <cassert>
#include "msg.h"
#include "util.h"
#include "fft.h"
#include "particle.h"

using namespace std;

Particles::Particles(const int nc, const double boxsize_) :
  np_local(0)
{
  size_t nx= fft_local_nx(nc);
  size_t np_alloc= (size_t)((1.25*(nx + 1)*nc*nc));

  pv= new vector<Particle>(np_alloc);
  p= &pv->front();
  //p= (Particle*) malloc(np_alloc*sizeof(Particle)); assert(p);
  force= (Float3*) calloc(3*np_alloc, sizeof(Float)); assert(force);
  np_allocated= np_alloc;
  boxsize= boxsize_;
  np_local= 0;
  np_total= 0;
  
  msg_printf(msg_verbose, "%lu Mbytes allocated for %lu particles\n",
	     mbytes(np_alloc*sizeof(Particle)), np_alloc);
}


Particles::~Particles()
{
  free(p);
}

