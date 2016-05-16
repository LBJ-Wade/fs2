#include "py_hdf5_io.h"

PyObject* py_hdf5_write_particles(PyObject* self, PyObject* args)
{
  // _hdf5_write_particles(_particles, filename)
  PyObject *bytes, *particles;

  if(!PyArg_ParseTuple(args, "O&O", PyUnicode_FSConverter, &bytes, py_particles)) {
    return NULL;
  }

  char* filename;
  Py_ssize_t len;
  PyBytes_AsStringAndSize(bytes, &filename, &len);

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  try {
    hdf5_write_particles(filename, particles, "xv");
  }
  catch(const IOError e) {
    Py_DECREF(bytes);
    PyErr_SetNone(PyExc_IOError);
    return NULL;
  }

  Py_DECREF(bytes);

  Py_RETURN_NONE;
}


