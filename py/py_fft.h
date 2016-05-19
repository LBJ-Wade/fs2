#ifndef PY_GRID_H
#define PY_GRID_H 1

#include "fft.h"
#include "Python.h"

PyMODINIT_FUNC
py_fft_module_init();

PyObject* fft_fx_as_array(FFT* const fft);

#endif
