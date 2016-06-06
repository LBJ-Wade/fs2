#ifndef PY_PM
#define PY_PM 1

#include "Python.h"

PyObject* py_pm_init(PyObject* self, PyObject* args);
PyObject* py_pm_compute_force(PyObject* self, PyObject* args);

PyObject* py_pm_compute_density(PyObject* self, PyObject* args);
PyObject* py_pm_domain_init(PyObject* self, PyObject* args);
PyObject* py_pm_send_positions(PyObject* self, PyObject* args);
PyObject* py_pm_write_packet_info(PyObject* self, PyObject* args);
#endif
