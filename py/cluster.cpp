// Cluster 

#include <iostream>
#include <cstdlib>
#include "msg.h"
#include <vector>
#include "particle.h"
#include "error.h"
#include "cluster.h"

using namespace std;

void cluster_init(Particles const * const particles, vector<Cluster>& v)
{
  v.reserve(particles->np_allocated);

  Cluster c;
  c.lbegin= c.lend= 0;

  const size_t n= particles->np_local;
  for(size_t i=0; i<n; ++i) {
    c.pbegin= i;
    c.pend= i+1;

    v.push_back(c);
  }
}
