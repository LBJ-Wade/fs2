#include "comm.h"
#include "py_fft.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

PyMODINIT_FUNC
py_fft_module_init()
{
  import_array();
  return NULL;
}

class FFTGridX {
 public:
  FFTGridX(FFT* const fft);
  ~FFTGridX();
 private:
  float_t* grid;
};

FFTGridX::FFTGridX(FFT* const fft) :
  grid(0)
{
  // call mpi gather
}

FFTGridX::~FFTGridX()
{
  free(grid);
}

PyObject* fft_fx_global_as_array(FFT* const fft)
{
  // return whole nc^3 fft->fx grid as an np.array at node 0
  // return None for node != 0
  const int nc= fft->nc;
  const int ncz= 2*(nc/2 + 1);
  const int nsend= fft->local_nx*nc*nc;
  float_t* const sendbuf= (float_t*) malloc(sizeof(float_t)*nsend);

  size_t i=0;
  for(size_t ix=0; ix<fft->local_nx; ++ix) 
   for(size_t iy=0; iy<nc; ++iy) 
    for(size_t iz=0; iz<nc; ++iz) 
      sendbuf[i++]= fft->fx[(nc*ix + iy)*ncz + iz];
  
  float_t* recvbuf;

  if(comm_n_nodes() == 1) {
    recvbuf= sendbuf;
  }
  else {
    const int nnodes= comm_n_nodes();

    int* nrecv= 0;

    if(comm_this_node() == 0) {
      // Allocate for gathered data
      nrecv= (int*) malloc(sizeof(int)*comm_n_nodes());
      if(nrecv == 0) {
	// ToDo: set error message
	return NULL;
      }

      recvbuf= (float_t*) malloc(sizeof(float_t)*nc*nc*nc);
      if(recvbuf == 0) {
	// sent string memory allocation error
	return NULL;
      }
    }
    
    MPI_Gather(&nsend, 1, MPI_INT, nrecv, nnodes, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Gatherv(sendbuf, nsend, FLOAT_TYPE, 
		recvbuf, nrecv, FLOAT_TYPE, 0, MPI_COMM_WORLD);

    free(nrecv);
    free(sendbuf);
  }

  if(comm_this_node() == 0) {
    //return  np array
    const int nd= 3;
    npy_intp dims[]= {nc, nc, nc};

    // ToDo; fix nrecv memory deallocation
    return PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, recvbuf);
  }

  Py_RETURN_NONE;
}

PyObject* fft_fx_as_array(FFT* const fft)
{
  // return fft->fx as an np.array
  const int nd= 3;
  const int nx= fft->local_nx;
  const int nc= fft->nc;
  const int ncz= 2*(nc/2 + 1);
  const npy_intp dims[]= {nx, nc, nc};
  const npy_intp fsize= sizeof(float_t);
  const npy_intp strides[]= {nx*ncz*fsize, ncz*fsize, fsize};

  return PyArray_New(&PyArray_Type, nd, dims, NPY_FLOAT, strides,
		     fft->fx, 0, 0, 0);
}


// How can I free a memory for PyArray? Give PyObject??
// ToDo: gather nc^3 grid to node 0
// return a PyArray with descructor?

