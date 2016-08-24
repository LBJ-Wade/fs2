#include <cmath>
#include "config.h"
#include "particle.h"
#include "comm.h"
#include "kdtree.h"
#include "fof.h"
#include "py_fof.h"
#include "py_array.h"
#include "py_assert.h"

using namespace std;


PyObject* py_fof_find_groups(PyObject* self, PyObject* args)
{
  // _fof_find_groups(_particles, linking_factor, quota)

  PyObject* py_particles;
  double linking_factor;
  int quota;
  if(!PyArg_ParseTuple(args, "Odi", &py_particles, &linking_factor, &quota))
    return NULL;

  py_assert_ptr(comm_n_nodes() == 1);
  
  Particles* const particles=
    (Particles*) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  const double linking_length= linking_factor
    * particles->boxsize/pow((double) particles->np_total, 1.0/3.0);
  
  
  fof_find_groups(particles, linking_length, quota);

  return py_vector_asarray<Index>(fof_nfof());
}

