#include <cmath>
#include "kdtree.h"
#include "py_assert.h"
#include "py_kdtree.h"

PyObject* py_kdtree_create_copy(PyObject* self, PyObject* args)
{
  // Copy the kdtree to Python data structure

  // size_t inode= 0;
  KdTree const * const kdtree= kdtree_get_root();
  size_t height= kdtree_get_height();
 

  size_t n_node= 2*pow(2, height) - 1;

  PyObject* const list= PyList_New(0);
  py_assert_ptr(list);
    
  for(size_t i=0; i<n_node; ++i) {
    KdTree const * node= kdtree + i;
    PyObject* item= Py_BuildValue("(KiddKK)",
				  (unsigned long long) i,
				  node->k,
				  (double) node->left,
				  (double) node->right,
				  (unsigned long long) node->ibegin,
				  (unsigned long long) node->iend);
    int ret= PyList_Append(list, item);
    py_assert_ptr(ret == 0);
  }
  
  return list;
}

PyObject* py_kdtree_get_height(PyObject* self, PyObject* args)
{
  return Py_BuildValue("K", (unsigned long long) kdtree_get_height());
}
