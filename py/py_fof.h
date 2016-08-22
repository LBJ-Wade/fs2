#ifndef PY_FOF_H
#define PY_FOF_H 1

#include "Python.h"
PyObject* py_fof_init(PyObject* self, PyObject* args);

PyMODINIT_FUNC
py_fof_module_init();

#endif
