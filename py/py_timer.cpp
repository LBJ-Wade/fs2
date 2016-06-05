#include "timer.h"
#include "error.h"
#include "py_timer.h"

PyObject* py_timer_save(PyObject* self, PyObject* args)
{
  // timer_save(filename)
  PyObject* bytes;
  char* filename;
  Py_ssize_t len;

  if(!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &bytes)) {
    return NULL;
  }

  PyBytes_AsStringAndSize(bytes, &filename, &len);


  try {
    timer_write(filename);
  }
  catch(IOError) {
    Py_DECREF(bytes);
    PyErr_SetNone(PyExc_IOError);
    return NULL;
  }

  Py_DECREF(bytes);

  Py_RETURN_NONE;
}
