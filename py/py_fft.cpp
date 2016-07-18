#include <iostream>
#include <mpi.h>
#include "config.h"
#include "comm.h"
#include "fft.h"
#include "py_assert.h"
#include "py_fft.h"

using namespace std;

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

#ifdef DOUBLEPRECISION
#define NPY_FLOAT_TYPE NPY_DOUBLE
#else
#define NPY_FLOAT_TYPE NPY_FLOAT
#endif


PyMODINIT_FUNC
py_fft_module_init()
{
  import_array();
  return NULL;
}

PyObject* py_fft_alloc(PyObject* self, PyObject* args)
{
  // _fft_alloc(nc)
  int nc;
  if(!PyArg_ParseTuple(args, "i", &nc)) {
    return NULL;
  }

  FFT* fft= 0;
  try {
    fft= new FFT("py_fft", nc, 0, true);
  }
  catch(MemoryError) {
    PyErr_SetNone(PyExc_MemoryError);
    return NULL;
  }

  return PyCapsule_New(fft, "_FFT", py_fft_free);
}

void py_fft_free(PyObject *obj)
{
  FFT* const fft=
    (FFT*) PyCapsule_GetPointer(obj, "_FFT");
  py_assert_void(fft);

  delete fft;
}

PyObject* py_fft_set_test_data(PyObject* self, PyObject* args)
{
  // _fft_set_test_data(_fft); set 1,2,3, ...
  PyObject* py_fft;
  if(!PyArg_ParseTuple(args, "O", &py_fft))
    return NULL;

  FFT* const fft=
      (FFT *) PyCapsule_GetPointer(py_fft, "_FFT");
  py_assert_ptr(fft);

  const size_t nc= fft->nc;
  const size_t ncz= 2*(nc/2 + 1);
  const size_t nx= fft->local_nx;
  const size_t ix0= fft->local_ix0;
  Float* const fx= fft->fx;
  
  for(size_t ix=0; ix<nx; ++ix) {
    for(size_t iy=0; iy<nc; ++iy) {
      for(size_t iz=0; iz<nc; ++iz) {
	size_t ilocal=  (ix*nc + iy)*ncz + iz;
	size_t iglobal= ((ix0 + ix)*nc + iy)*nc + iz;
	fx[ilocal]= iglobal + 1;
      }
    }
  }

  Py_RETURN_NONE;
}

PyObject* py_fft_fx_global_as_array(PyObject* self, PyObject* args)
{
  // _fft_fx_global_as_array(_fft)
  //   return whole nc^3 fft->fx grid as an np.array at node 0
  //   return None for node != 0

  PyObject* py_fft;
  if(!PyArg_ParseTuple(args, "O", &py_fft))
    return NULL;

  FFT* const fft=
      (FFT *) PyCapsule_GetPointer(py_fft, "_FFT");
  py_assert_ptr(fft);

  const int nc= fft->nc;
  const size_t ncz= 2*(nc/2 + 1);
  const size_t nx= fft->local_nx;
  
  //
  // Allocate a new np.array
  //
  PyObject* arr= 0;
  Float* recvbuf= 0;
    
  if(comm_this_node() == 0) {
    const int nd= 3;
    npy_intp dims[]= {nc, nc, nc};

    arr= PyArray_ZEROS(nd, dims, NPY_FLOAT_TYPE, 0);
    py_assert_ptr(arr);
  
    recvbuf= (Float*) PyArray_DATA((PyArrayObject*) arr);
    py_assert_ptr(recvbuf);
  }

  if(comm_n_nodes() == 1) {
    size_t i=0;
    for(size_t ix=0; ix<nx; ++ix)
     for(size_t iy=0; iy<nc; ++iy) 
      for(size_t iz=0; iz<nc; ++iz)
	recvbuf[i++]= fft->fx[(nc*ix + iy)*ncz + iz];
    
    return arr;
  }


  //
  // Gather global grid data if n_nodes > 1
  //
  const int nsend= fft->local_nx*nc*nc;

  const int n= comm_n_nodes();
  Float* const sendbuf= (Float*) malloc(sizeof(Float)*nsend);
  if(sendbuf == 0) {
    PyErr_SetString(PyExc_MemoryError,
		    "Unable to allocate memory for nrecv");
    return NULL;
  }

  size_t i=0;
  for(size_t ix=0; ix<nx; ++ix) 
    for(size_t iy=0; iy<nc; ++iy) 
      for(size_t iz=0; iz<nc; ++iz) 
	sendbuf[i++]= fft->fx[(nc*ix + iy)*ncz + iz];

  int* nrecv= 0;
  int* displ=0;

  if(comm_this_node() == 0) {

    // Allocate for gathered data
    nrecv= (int*) malloc(sizeof(int)*2*n);
    if(nrecv == 0) {
      PyErr_SetString(PyExc_MemoryError,
		      "Unable to allocate memory for nrecv");
      return NULL;
    }

    displ= nrecv + n;
  }
    
  MPI_Gather(&nsend, 1, MPI_INT, nrecv, 1, MPI_INT, 0, MPI_COMM_WORLD);
  if(comm_this_node() == 0) {
    int displ_sum= 0;
    for(int i=0; i<n; ++i) {
      displ[i]= displ_sum;
      displ_sum += nrecv[i];
    }
  }
    
  MPI_Gatherv(sendbuf, nsend, FLOAT_TYPE, 
	      recvbuf, nrecv, displ, FLOAT_TYPE, 0, MPI_COMM_WORLD);

  free(nrecv);
  free(sendbuf);

  if(comm_this_node() == 0) {
    return arr;
  }

  Py_RETURN_NONE;
}

/*
PyObject* py_fft_fx_as_array(FFT* const fft)
{
  // return fft->fx as an np.array
  const int nd= 3;
  const int nx= fft->local_nx;
  const int nc= fft->nc;
  const int ncz= 2*(nc/2 + 1);
  const npy_intp dims[]= {nx, nc, nc};
  const npy_intp fsize= sizeof(Float);
  const npy_intp strides[]= {nx*ncz*fsize, ncz*fsize, fsize};

  return PyArray_New(&PyArray_Type, nd, dims, NPY_FLOAT, strides,
		     fft->fx, 0, 0, 0);
}
*/

