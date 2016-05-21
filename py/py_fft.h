#ifndef PY_GRID_H
#define PY_GRID_H 1

#include "fft.h"
#include "Python.h"

PyMODINIT_FUNC
py_fft_module_init();

PyObject* py_fft_alloc(PyObject* self, PyObject* args);
void py_fft_free(PyObject *obj);
PyObject* py_fft_set_test_data(PyObject* self, PyObject* args);

//PyObject* py_fft_fx_as_array(FFT* const fft);
PyObject* py_fft_fx_global_as_array(PyObject* self, PyObject* args);


#endif
