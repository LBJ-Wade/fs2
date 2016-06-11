#ifndef PY_STAT_H
#define PY_STAT_H 1

#include "Python.h"

PyObject* py_stat_set_filename(PyObject* self, PyObject* args);
PyObject* py_stat_record_pm_nbuf(PyObject* self, PyObject* args);

#endif
