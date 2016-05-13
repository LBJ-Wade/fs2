#include "error.h"
#include "gadget_file.h"
#include "py_assert.h"
#include "py_write.h"

PyObject* py_write_gadget_binary(PyObject* self, PyObject* args)
{
  // _py_write_gadget_binary(_particles, filename)
  PyObject *py_particles, *bytes;
  char* filename;
  Py_ssize_t len;
  int use_long_id;

  if(!PyArg_ParseTuple(args, "OO&d",
		 &py_particles, PyUnicode_FSConverter, &bytes, &use_long_id)) {
    return NULL;
  }

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  PyBytes_AsStringAndSize(bytes, &filename, &len);

  try {
    gadget_file_write_particles(filename, particles, use_long_id);
  }
  catch(IOError) {
    Py_DECREF(bytes);
    PyErr_SetNone(PyExc_IOError);
    return NULL;
  }

  Py_DECREF(bytes);

  Py_RETURN_NONE;
}

