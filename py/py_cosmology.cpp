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

PyObject* py_cosmology_D_growth(PyObject* self, PyObject* args)
{
  double a;
  if(!PyArg_ParseTuple(args, "d", &a)) {
    return NULL;
  }

  return Py_BuildValue("d", cosmology_D_growth(a));
}

PyObject* py_cosmology_D2_growth(PyObject* self, PyObject* args)
{
  double a;
  if(!PyArg_ParseTuple(args, "d", &a)) {
    return NULL;
  }

  const double D1= cosmology_D_growth(a);
  const double D2= cosmology_D2_growth(a, D1);
  return Py_BuildValue("d", D2);
}

