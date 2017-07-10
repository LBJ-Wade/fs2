#ifndef PY_COSMOLOGY_H
#define PY_COSMOLOGY_H 1

#include "Python.h"

PyObject* py_cosmology_init(PyObject* self, PyObject* args);
PyObject* py_cosmology_D_growth(PyObject* self, PyObject* args);
PyObject* py_cosmology_D2_growth(PyObject* self, PyObject* args);

#endif
