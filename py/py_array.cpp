#include "py_array.h"
#include "py_error.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

PyMODINIT_FUNC
py_array_module_init()
{
  import_array();

  return NULL;
}

static int npy_type_num(const type_info& type_id)
{
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

template <class T>
PyObject* py_vector_asarray(vector<T>& v)
{
  const int typenum= npy_type_num(typeid(T));
  npy_intp dim= v.size();
  PyObject* const arr=
    PyArray_SimpleNewFromData(1, &npy_intp, typenum, &*v.front());

  return arr;
}
