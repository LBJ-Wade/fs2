#include "cola.h"
#include "py_assert.h"

PyObject* py_cola_kick(PyObject* self, PyObject* args)
{
  // _cola_kick(_particles, a_vel)

  PyObject *py_particles;
  double a_vel;
  
  if(!PyArg_ParseTuple(args, "Od", &py_particles, &a_vel)) {
    return NULL;
  }

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  cola_kick(particles, a_vel);

  Py_RETURN_NONE;
}

PyObject* py_cola_drift(PyObject* self, PyObject* args)
{
  // _cola_drift(_particles, a_pos)

  PyObject *py_particles;
  double a_pos;
  
  if(!PyArg_ParseTuple(args, "Od", &py_particles, &a_pos)) {
    return NULL;
  }

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  cola_drift(particles, a_pos);

  Py_RETURN_NONE;
}
