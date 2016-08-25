//
// Helper function to handle np.array
//

#include <iostream>
#include "py_array.h"
#include "msg.h"
#include "error.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

using namespace std;

PyMODINIT_FUNC
py_array_module_init()
{
  msg_printf(msg_debug, "Module py_array initialised for numpy array.");
  import_array();

  return NULL;
}

static int npy_type_num(const type_info& type_id)
{
  // Return NPY type corresponds to the C/C++ type
  // e.g.: npy_type_num(typeid(T))
  
  if(type_id == typeid(float))
    return NPY_FLOAT;
  else if(type_id == typeid(double))
    return NPY_DOUBLE;
  else if(type_id == typeid(int))
    return NPY_INT;
  else if(type_id == typeid(long))
    return NPY_LONG;
  else if(type_id == typeid(unsigned int))
    return NPY_UINT;
  else if(type_id == typeid(unsigned long))
    return NPY_ULONG;
  else if(type_id == typeid(uint64_t))
    return NPY_UINT64;
  else {
    msg_printf(msg_fatal, "Error: unknown typeid for npy_num_num");
    throw RuntimeError();
  }

  return 0;
}

template <typename T>
PyObject* py_vector_asarray(vector<T>& v)
{
  // Wrap numerical vector with np.array
  npy_intp dim= v.size();
  const int typenum= npy_type_num(typeid(T));
  PyObject* arr= PyArray_SimpleNewFromData(1, &dim, typenum, &v.front());

  return arr;
}

template PyObject* py_vector_asarray<Index>(vector<Index>& v);

