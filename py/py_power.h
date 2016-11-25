#ifndef PY_POWER_H
#define PY_POWER_H 1

#include "Python.h"

PyObject* py_power_alloc(PyObject* self, PyObject* args);
PyObject* py_power_sigma(PyObject* self, PyObject* args);
PyObject* py_power_n(PyObject* self, PyObject* args);
PyObject* py_power_ki(PyObject* self, PyObject* args);
PyObject* py_power_Pi(PyObject* self, PyObject* args);
PyObject* py_power_i(PyObject* self, PyObject* args);

#endif
