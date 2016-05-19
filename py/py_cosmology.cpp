#include "py_cosmology.h"
#include "cosmology.h"

PyObject* py_cosmology_init(PyObject* self, PyObject* args)
{
  // _cosmology_init(omega_m0, h)
  double omega_m, h;

  if(!PyArg_ParseTuple(args, "dd", &omega_m, &h)) {
    return NULL;
  }

  cosmology_init(omega_m, h);

  Py_RETURN_NONE;
}
