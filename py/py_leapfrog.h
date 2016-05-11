#ifndef PY_LEAPFROG_H
#define PY_LEAPFROG_H 1

#include "Python.h"

PyObject* py_leapfrog_initial_velocity(PyObject* self, PyObject* args);
PyObject* py_leapfrog_kick(PyObject* self, PyObject* args);
PyObject* py_leapfrog_drift(PyObject* self, PyObject* args);

#endif
