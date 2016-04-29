#ifndef PY_COMM_H
#define PY_COMM_H 1

#include "Python.h"

PyObject* py_comm_mpi_init(PyObject *self, PyObject* args);
PyObject* py_comm_mpi_finalise(PyObject *self, PyObject* args);
PyObject* py_comm_hello(PyObject *self, PyObject* args);

#endif
