#ifndef PY_ARRAY_H
#define PY_ARRAY_H 1

#include "Python.h"
#include <vector>

PyMODINIT_FUNC
py_array_module_init();

template <class T> PyObject* py_vector_asarray(std::vector<T>& v);


#endif
