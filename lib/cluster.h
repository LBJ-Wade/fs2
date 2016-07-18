#ifndef CLUSTER_H
#define CLUSTER_H 1

struct Edge {
  int i,j;
};

struct Cluster {
  int pbegin, pend;
  int lbegin, lend;
};

#endif
