#include "leapfrog.h"
#include "py_assert.h"
#include "py_leapfrog.h"

PyObject* py_leapfrog_initial_velocity(PyObject* self, PyObject* args)
{
  PyObject *py_particles;
  double a_vel;
  
  if(!PyArg_ParseTuple(args, "Od", &py_particles, &a_vel)) {
    return NULL;
  }

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  leapfrog_set_initial_velocity(particles, a_vel);

  Py_RETURN_NONE;
}

PyObject* py_leapfrog_kick(PyObject* self, PyObject* args)
{
  // _leapfrog_kick(_particles, a_vel)

  PyObject *py_particles;
  double a_vel;
  
  if(!PyArg_ParseTuple(args, "Od", &py_particles, &a_vel)) {
    return NULL;
  }

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  leapfrog_kick(particles, a_vel);

  Py_RETURN_NONE;
}

PyObject* py_leapfrog_drift(PyObject* self, PyObject* args)
{
  // _leapfrog_drift(_particles, a_pos)

  PyObject *py_particles;
  double a_pos;
  
  if(!PyArg_ParseTuple(args, "Od", &py_particles, &a_pos)) {
    return NULL;
  }

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  leapfrog_drift(particles, a_pos);

  Py_RETURN_NONE;
}




