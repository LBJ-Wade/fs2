#include "config.h"
#include "py_config.h"

#ifndef PRECISION
#error "Error: PRECISION undefined"
#endif

PyObject* py_config_precision(PyObject* self, PyObject* args)
{
  // return "single" or "double"
  return Py_BuildValue("s", PRECISION);
}
