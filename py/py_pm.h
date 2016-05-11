#ifndef PY_PM
#define PY_PM 1

#include "Python.h"

PyObject* py_pm_init(PyObject* self, PyObject* args);
PyObject* py_pm_compute_force(PyObject* self, PyObject* args);

#endif
