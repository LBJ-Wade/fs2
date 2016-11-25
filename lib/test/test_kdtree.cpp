#include <cassert>
#include "fs.h"

using namespace std;

static void test_kdtree(Particles* const particles, KdTree const * const kdtree);

int main(int argc, char* argv[])
{
  comm_mpi_init(&argc, &argv);
  
  const double omega_m= 0.308;
  const int nc= 64;
  const Float boxsize= 64.0;
  const unsigned long seed= 1;
  const double a_final= 1.0;

  msg_set_loglevel(msg_info);
  cosmology_init(omega_m);

  PowerSpectrum* ps= new PowerSpectrum("../../data/planck_matterpower.dat");
  Particles* particles= new Particles(nc, boxsize);

  lpt_init(nc, boxsize, 0);
  lpt_set_displacements(seed, ps, a_final, particles);

  // ToDo: This boxsize3 is for 1 node
  Float boxsize3[]= {boxsize, boxsize, boxsize};
  KdTree const * const kdtree= kdtree_init(particles, boxsize3);

  test_kdtree(particles, kdtree);

  lpt_free();
  delete ps;
  delete particles;
  comm_mpi_finalise();
  
  return 0;
}



KdTree const * search_nearest_leaf(const Float x[], KdTree const * const kdtree, const size_t inode)
{
  // Find a leaf that contains a particle position x
  KdTree const * const tree= kdtree + inode;
  if(tree->iend - tree->ibegin <= KdTree::quota)
    return tree;
  
  const size_t ileft= left_child(inode);
  KdTree const * const l= kdtree + ileft;

  if(l->left <= x[l->k] && x[l->k] <= l->right)
    return search_nearest_leaf(x, kdtree, ileft);

  const size_t iright= right_child(inode);
  KdTree const * const r= kdtree + iright;  

  if(r->left <= x[r->k] && x[r->k] <= r->right)
    return search_nearest_leaf(x, kdtree, iright);

  assert(false);
  return 0;
}



void test_kdtree(Particles* const particles, KdTree const * const kdtree)
{
  Particle const * const p= particles->p;

  size_t n= 0;
  
  for(size_t i=0; i<particles->np_local; ++i) {
    KdTree const * const tree= search_nearest_leaf(p[i].x, kdtree, 0);
    if(!(tree->ibegin <= i && i < tree->iend)) {
      msg_abort("Error: test_kdtree failed for particle %ul\n", i);
    }
    n++;
  }

  msg_printf(msg_info, "test_kdtree sucessful for %lu particles.\n", n);
  
}
