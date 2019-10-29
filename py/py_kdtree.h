#ifndef PY_KDTREE_H
#define PY_KDTREE_H 1

#include "Python.h"

PyObject* py_kdtree_create_copy(PyObject* self, PyObject* args);
PyObject* py_kdtree_get_height(PyObject* self, PyObject* args);

#endif

