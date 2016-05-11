#include "py_cosmology.h"
#include "cosmology.h"

PyObject* py_cosmology_init(PyObject* self, PyObject* args)
{
  // _cosmology_init(omega_m0)
  double omega_m;

  if(!PyArg_ParseTuple(args, "d", &omega_m)) {
    return NULL;
  }

  cosmology_init(omega_m);

  Py_RETURN_NONE;
}
