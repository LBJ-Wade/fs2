#include <iostream>
#include <vector>
#include <queue>
#include <cassert>

#include "msg.h"
#include "kdtree.h"

using namespace std;

static KdTree const * kdtree;
static Float boxsize= 0, half_boxsize;
static Float ll, ll2;
static vector<Index> grp;
static vector<Index> nfof;
static queue<Index> q;
static Particle const * p;
static size_t n_reserve= 0;

static inline Float periodic_dx(Float dx)
{
  dx= dx < -half_boxsize ? dx + boxsize : dx;
  dx= dx >= half_boxsize ? dx - boxsize : dx;

  return dx;
}

static inline Float dist2(const Float x[], const Float y[])
{
  // 3D Euclidian distance with periodic boundary condition
  Float dx= periodic_dx(x[0] - y[0]);
  Float dy= periodic_dx(x[1] - y[1]);
  Float dz= periodic_dx(x[2] - y[2]);

  return dx*dx + dy*dy + dz*dz;
}

static void link_particles_recursive(const size_t inode, const Index i);



void fof_find_groups(Particles* const particles,
		     const Float linking_length,
		     Float const * const boxsize3,
		     const int quota)
{
  // Apply Friends-of-Friends (FoF) halo finder on particles
  // A pair of particles x and y will be in the same member if
  // |x - y| < linking_length
  // particles will be suffled by kdtree_init()

  boxsize= particles->boxsize;
  half_boxsize= particles->boxsize / 2;
  ll= linking_length;
  ll2= linking_length*linking_length;
  assert(q.empty());

  msg_printf(msg_info, "Linking length %f\n", ll);

  // grp[i] is the group number that particle i belongs to
  n_reserve= particles->np_allocated;
  grp.clear();
  grp.reserve(n_reserve);

  // nfof[i] is a vector of number of FoF member particles for
  //nfof.clear();
  //nfof.reserve(particles->np_allocated);

  // Each particles is a group
  const Index n= particles->np_local;
  for(Index i=0; i<n; ++i)
    grp.push_back(i);

  kdtree= kdtree_init(particles, boxsize3, quota);
  
  p= particles->p;

  Index ngrp= 0;

  for(Index i=0; i<n; ++i) {
    if(grp[i] != i)
      continue; // already belongs to other group

    // Start finding a new FoF group
    ngrp++;
    assert(q.empty());
    q.push(i);

    while(!q.empty()) {
      // Find friends of particle q.front
      link_particles_recursive(0, q.front());
      q.pop();
    }

    //nfof.push_back(nfriends);
  }

  //msg_printf(msg_info, "fof %lu groups found.\n", (unsigned long) nfof.size());
}

vector<Index>& fof_compute_nfof()
{
  // nfof[i] is a vector of number of FoF member particles for
  nfof.clear();
  nfof.reserve(n_reserve);
  nfof.assign(grp.size(), 0);

  const Index n= grp.size();
  for(Index i=0; i<n; ++i) {
    nfof[grp[i]]++;
  }

  return nfof;
}



size_t fof_ngroups()
{
  return nfof.size();
}

vector<Index>& fof_nfof()
{
  return nfof;
}

vector<Index>& fof_grp()
{
  return grp;
}



void link_particles_recursive(const size_t inode, const Index i)
{
  // Find particles near particle i and add to queue if that particle
  // is prestine

  KdTree const * const tree= kdtree + inode;
  const int k= kdtree->k;
  Float x= p[i].x[k];

  if((x < tree->left  - ll && x + boxsize > tree->right + ll) ||
     (x > tree->right + ll && x - boxsize < tree->left  - ll)) {

    return; // This node is far enough from particle i
  }

  // Add neightbor particles if inode is a leaf
  if(tree->iend - tree->ibegin <= KdTree::quota) {
    for(Index j=tree->ibegin; j<tree->iend; ++j) {
      if(grp[j] != grp[i] && dist2(p[i].x, p[j].x) < ll2) {
	q.push(j);
	grp[j]= grp[i];
      }
    }
    return;
  }

  // Search subtrees recursively
  link_particles_recursive(left_child(inode), i);
  link_particles_recursive(right_child(inode), i);
}

