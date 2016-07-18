#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "config.h"
#include "msg.h"
#include "error.h"
#include "util.h"
#include "kdtree.h"


using namespace std;

static KdTree* kdtree= 0;
static size_t ntree_alloc= 0;

template<typename T> class CompPoints {
 public:
  CompPoints(const int direction) : k(direction) {}
  bool operator()(const T& p, const T& q) const {
    return p.x[k] < q.x[k];
  }
 private:
  int k;
};

static inline int cut_direction(const Float boxsize[])
{
  int k=1;
  if(boxsize[2] > boxsize[1])
    k=2;
  if(boxsize[0] > boxsize[k])
    k=0;

  return k;
}


template<typename T>
void construct_recursive(vector<T>& v, const size_t inode, 
			 const size_t ibegin, const size_t iend,
			 Float left[], Float right[], Float boxsize[])
{
  assert(iend - ibegin >= 2);
  assert(0 <= inode && inode < v.size() - 1);
  KdTree* const tree= kdtree + inode;
  
  const int k= cut_direction(boxsize);
  tree->direction= k;
  boxsize[k] /= 2;

  const size_t imid= ibegin + (iend - ibegin)/2;
  nth_element(v.begin() + ibegin, v.begin() + imid, v.begin() + iend,
	      CompPoints<T>(k));

  if(imid - ibegin >= 2) {
    construct_recursive(v, inode+1, ibegin, imid, left, right,
			boxsize);
  }
  else {
    left[0]= right[0]= v[ibegin].x[0];
    left[1]= right[1]= v[ibegin].x[1];
    left[2]= right[2]= v[ibegin].x[2];
  }

  Float left1[3], right1[3];
  
  if(iend - imid >= 2) {
    Float left1[3];
    construct_recursive(v, inode + (imid - ibegin), imid, iend, left1, right1,
			boxsize);
  }
  else {
    left1[0]= right1[0]= v[imid].x[0];
    left1[1]= right1[1]= v[imid].x[1];
    left1[2]= right1[2]= v[imid].x[2];
  }

  left[0]= min(left[0], left1[0]);
  left[1]= min(left[1], left1[1]);
  left[2]= min(left[2], left1[2]);

  right[0]= max(right[0], right1[0]);
  right[1]= max(right[1], right1[1]);
  right[2]= max(right[2], right1[2]);

  tree->left= left[k];    
  tree->right= right[k];
}



//
// Global functions
//
void kdtree_init(Particles* const particles, const Float boxsize[])
{
  ntree_alloc= particles->np_allocated - 1;
  kdtree= (KdTree*) malloc(sizeof(KdTree)*ntree_alloc);
  size_t mem_size= mbytes(sizeof(KdTree)*ntree_alloc);

  if(kdtree) {
    msg_printf(msg_info, "Allocated %lu Mbytes for kdtree\n", mem_size);
  }
  else {	       
    msg_printf(msg_fatal, "Error: unable to allocate memory for %lu kdtrees, "
	       "%lu mbytes required\n", ntree_alloc, mem_size);
    throw MemoryError();
  }

  Float left[3], right[3];
  Float boxsize1[]= {boxsize[0], boxsize[1], boxsize[2]};

  construct_recursive(*particles->pv, 0, 0, particles->np_local,
		      left, right, boxsize1);
}





