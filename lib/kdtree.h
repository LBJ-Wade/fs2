#ifndef KDTREE_H
#define KDTREE_H 1

#include <vector>
#include "config.h"
#include "cluster.h"

struct KdTree {
  int direction;
  Float left, right;
};

void kdtree_construct(std::vector<Cluster>& v, const Float boxsize[]);

#endif
