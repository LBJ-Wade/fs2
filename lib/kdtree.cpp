#include <iostream>
#include <vector>
#include <queue>
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
static size_t height;

int KdTree::quota;


//static void traverse_tree_recursive(const size_t inode);

static inline void set_left_right(KdTree * const tree, const int k,
			     const Float left[], const Float right[])
{
  tree->left= left[k];
  tree->right= right[k];
  /*
  tree->left[0]= left[0];
  tree->left[1]= left[1];
  tree->left[2]= left[2];
  tree->right[0]= right[0];
  tree->right[1]= right[1];
  tree->right[2]= right[2];
  */
}


template<typename T> class CompPoints {
 public:
  CompPoints(const int direction) : k(direction) {}
  bool operator()(const T& p, const T& q) const {
    return p.x[k] < q.x[k];
  }
 private:
  int k;
};

static inline int cut_direction(const Float boxsize3[])
{
  // Find the longest edge of a cuboid
  int k=1;
  if(boxsize3[2] > boxsize3[1])
    k=2;
  if(boxsize3[0] > boxsize3[k])
    k=0;

  return k;
}




template<typename T>
void compute_bounding_box(const vector<T>& v, 
			  const size_t ibegin, const size_t iend,
			  Float left[], Float right[])
{
  left[0]= right[0]= v[ibegin].x[0];
  left[1]= right[1]= v[ibegin].x[1];
  left[2]= right[2]= v[ibegin].x[2];

  for(size_t i=ibegin+1; i<iend; ++i) {
    left[0]= min(left[0], v[i].x[0]);
    left[1]= min(left[1], v[i].x[1]);
    left[2]= min(left[2], v[i].x[2]);

    right[0]= max(right[0], v[i].x[0]);
    right[1]= max(right[1], v[i].x[1]);
    right[2]= max(right[2], v[i].x[2]);
  }
}

template<typename T>
void construct_recursive_balanced(vector<T>& v, const size_t inode, 
			 const Index ibegin, const Index iend,
			 Float left[], Float right[], Float boxsize3[])
{
  assert(0 <= inode && inode < ntree_alloc);
  KdTree* const tree= kdtree + inode;

  tree->ibegin= ibegin;
  tree->iend= iend;
  
  if(iend - ibegin <= KdTree::quota) {
    compute_bounding_box(v, ibegin, iend, left, right);
    set_left_right(tree, tree->k, left, right);
    return;
  }
  
  const int k= cut_direction(boxsize3);
  boxsize3[k] /= 2;

  const size_t imid= ibegin + (iend - ibegin)/2;
  nth_element(v.begin() + ibegin, v.begin() + imid, v.begin() + iend,
	      CompPoints<T>(k));

  const size_t ileft= left_child(inode);
  kdtree[ileft].k= k;
  construct_recursive_balanced(v, ileft, ibegin, imid, left, right, boxsize3);
  set_left_right(kdtree + ileft, k, left, right);

  Float left1[3], right1[3];
  const size_t iright= right_child(inode);
  kdtree[iright].k= k;
  construct_recursive_balanced(v, iright, imid, iend, left1, right1, boxsize3);
  set_left_right(kdtree + iright, k, left1, right1);

  left[0]= min(left[0], left1[0]);
  left[1]= min(left[1], left1[1]);
  left[2]= min(left[2], left1[2]);

  right[0]= max(right[0], right1[0]);
  right[1]= max(right[1], right1[1]);
  right[2]= max(right[2], right1[2]);
}




  
//
// Global functions
//
KdTree* kdtree_init(Particles* const particles,
		    Float const * const boxsize3, const int quota)
{
  // boxsize3: the size of the cuboid containing particles. This is only used
  //           to determine the direction of cut, does not have to be an exact
  //           bounding box
  // quota (optional): the maximum number of particles in the leaf.
  //                   Empirically, default 32 is OK.
  
  const size_t np= particles->np_allocated;
  Index nleaf= 1;
  int height_new= 0;
  while(quota*nleaf < np) {
    nleaf= nleaf << 1;
    height_new++;
  }

  assert(nleaf == (1 << height_new));

  KdTree::quota= quota;
  
  size_t ntree_alloc_new= 2*nleaf - 1;

  if(ntree_alloc_new > ntree_alloc) {
    // Allocate (more) memory for KdTree
    ntree_alloc= ntree_alloc_new;
    height= height_new;
    
    free(kdtree);
    kdtree= (KdTree*) malloc(sizeof(KdTree)*ntree_alloc);
    size_t mem_size= mbytes(sizeof(KdTree)*ntree_alloc);
    
  
    if(kdtree) {
      msg_printf(msg_info, "Allocated %lu Mbytes for kdtree\n", mem_size);
      msg_printf(msg_verbose, "Tree depth= %d, leaves= %d\n", height, nleaf);
    }
    else {	       
      msg_printf(msg_fatal, "Error: unable to allocate memory for %lu kdtrees, "
		 "%lu mbytes required\n", ntree_alloc, mem_size);
      throw MemoryError();
    }
  }

  Float left[3], right[3];
  Float boxsize3_copy[3];
  if(boxsize3) {
    boxsize3_copy[0]= boxsize3[0];
    boxsize3_copy[1]= boxsize3[1];
    boxsize3_copy[2]= boxsize3[2];
  }
  else {
    compute_bounding_box(*particles->pv, 0, particles->np_local, left, right);
    boxsize3_copy[0]= right[0] - left[0];
    boxsize3_copy[1]= right[1] - left[1];
    boxsize3_copy[2]= right[2] - left[2];
  }

  construct_recursive_balanced(*particles->pv, 0, 0, particles->np_local,
			       left, right, boxsize3_copy);

  kdtree->k= 0;
  set_left_right(kdtree, 0, left, right);

  //traverse_tree_recursive(0); // !!! debug
  
  return kdtree;
}

/*
void traverse_tree_recursive(const size_t inode)
{
  KdTree* const tree= kdtree + inode;
  fprintf(stderr, "%zu %d [%f %f]\n", inode, tree->k, tree->left, tree->right);

  if(is_leaf(tree)) return;

  traverse_tree_recursive(left_child(inode));
  traverse_tree_recursive(right_child(inode));
}
*/
