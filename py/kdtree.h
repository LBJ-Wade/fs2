#ifndef KDTREE_H
#define KDTREE_H 1

#include <vector>
#include "config.h"
#include "particle.h"
//#include "cluster.h"

struct KdTree {
  int k;
  //Float left[3], right[3];
  Float left, right;
  Index ibegin, iend;
  static int quota;
};

static inline size_t left_child(const size_t i)
{
  return (i << 1) + 1;
}

static inline size_t right_child(const size_t i)
{
  return (i << 1) + 2;
}

static inline bool is_leaf(KdTree const * const tree)
{
  return (tree->iend - tree->ibegin) < KdTree::quota;
}


//void kdtree_construct(std::vector<Cluster>& v, const Float boxsize[]);
KdTree* kdtree_init(Particles* const particles, const Float boxsize[], const int quota_=32);

KdTree* kdtree_get_root();
size_t kdtree_get_height();

#endif
