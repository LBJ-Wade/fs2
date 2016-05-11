#ifndef PY_PARTICLES_H
#define PY_PARTICLES_H 1

#include "Python.h"

PyMODINIT_FUNC
py_particles_module_init();

PyObject* py_particles_alloc(PyObject* self, PyObject* args);
void py_particles_free(PyObject *obj);
PyObject* py_particles_len(PyObject* self, PyObject* args);
PyObject* py_particles_slice(PyObject* self, PyObject* args);
PyObject* py_particles_getitem(PyObject* self, PyObject* args);

#endif
