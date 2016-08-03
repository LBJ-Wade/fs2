#include "mem.h"
#include "error.h"
#include "py_assert.h"
#include "py_mem.h"


static void py_mem_free(PyObject *obj);

PyObject* py_mem_alloc(PyObject* self, PyObject* args)
{
  // _mem_alloc(name, size)
  const char* name;
  unsigned long long size;

  if(!PyArg_ParseTuple(args, "sK", &name, &size)) {
    return NULL;
  }

  Mem* mem= 0;
  if(size == 0)
    mem= new Mem(name);
  else {
    try {
      mem= new Mem(name, size);
    }
    catch(MemoryError e) {
      PyErr_SetNone(PyExc_MemoryError);
      return NULL;
    }
  }

  return PyCapsule_New(mem, "_Mem", py_mem_free);
}

void py_mem_free(PyObject *obj)
{
  Mem* const mem= (Mem*) PyCapsule_GetPointer(obj, "_Mem");
  py_assert_void(mem);

  delete mem;
}


PyObject* py_mem(PyObject* self, PyObject* args)
{
  PyObject* py_mem;
  if(!PyArg_ParseTuple(args, "O", &py_mem)) {
    return NULL;
  }

  Mem* const mem= (Mem*) PyCapsule_GetPointer(py_mem, "_Mem");
  py_assert_ptr(mem);

  return Py_BuildValue("(ii)", mem->size_alloc, mem->size_using);
}
