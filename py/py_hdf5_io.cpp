#include "hdf5_io.h"
#include "error.h"
#include "py_hdf5_io.h"
#include "py_assert.h"

PyObject* py_hdf5_write_particles(PyObject* self, PyObject* args)
{
  // _hdf5_write_particles(_particles, filename, "ixvf")
  PyObject *bytes, *py_particles, *py_ixvf;

  if(!PyArg_ParseTuple(args, "OO&O&", &py_particles,
		       PyUnicode_FSConverter, &bytes,
		       PyUnicode_FSConverter, &py_ixvf)) {
    return NULL;
  }

  char* filename;
  Py_ssize_t len;
  PyBytes_AsStringAndSize(bytes, &filename, &len);

  // ToDo Check how to get ascii char only ixvf12 expected
  char* ixvf;
  Py_ssize_t len_ixvf;
  PyBytes_AsStringAndSize(py_ixvf, &ixvf, &len_ixvf);

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  try {
    hdf5_write_particles(filename, particles, ixvf);
  }
  catch(const IOError e) {
    Py_DECREF(bytes);
    Py_DECREF(py_ixvf);
    PyErr_SetNone(PyExc_IOError);
    return NULL;
  }

  Py_DECREF(bytes);
  Py_DECREF(py_ixvf);

  Py_RETURN_NONE;
}


