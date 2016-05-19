#include "py_fft.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

PyMODINIT_FUNC
py_fft_module_init()
{
  import_array();
  return NULL;
}

PyObject* fft_fx_as_array(FFT* const fft)
{
  const int nd= 3;
  const int nc= fft->nc;
  const int ncz= 2*(nc/2 + 1);
  npy_intp dims[]= {nc, nc, nc};
  npy_intp fsize= sizeof(float_t);
  npy_intp strides[]= {nc*ncz*fsize, ncz*fsize, fsize};

  return PyArray_New(&PyArray_Type, nd, dims, NPY_FLOAT, strides,
		     fft->fx, 0, 0, 0);
  //return PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, fft->fx);
}
