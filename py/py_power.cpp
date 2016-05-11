//
// wrapping power.cpp
//
#include "Python.h"

#include "power.h"
#include "py_power.h"
#include "py_assert.h"

using namespace std;

static void py_power_free(PyObject *obj);

PyObject* py_power_alloc(PyObject* self, PyObject* args)
{
  // _power_alloc(filename)
  PyObject* bytes;
  char* filename;
  Py_ssize_t len;

  if(!PyArg_ParseTuple(args, "O&", PyUnicode_FSConverter, &bytes)) {
    return NULL;
  }

  PyBytes_AsStringAndSize(bytes, &filename, &len);

  PowerSpectrum* ps;

  try {
    ps= new PowerSpectrum(filename);
  }
  catch(PowerFileError) {
    Py_DECREF(bytes);
    PyErr_SetString(PyExc_IOError, "PowerFileError");
    return NULL;
  }

  Py_DECREF(bytes);

  return PyCapsule_New(ps, "_PowerSpectrum", py_power_free);
}


void py_power_free(PyObject *obj)
{
  PowerSpectrum* const ps=
    (PowerSpectrum*) PyCapsule_GetPointer(obj, "_PowerSpectrum");
  py_assert_void(ps);

  delete ps;
}

PyObject* py_power_sigma(PyObject* self, PyObject* args)
{
  PyObject* py_ps;
  double R;
  
  if(!PyArg_ParseTuple(args, "Od", &py_ps, &R))
    return NULL;

  PowerSpectrum* const ps=
    (PowerSpectrum *) PyCapsule_GetPointer(py_ps, "_PowerSpectrum");
  py_assert_ptr(ps);

  return Py_BuildValue("d", ps->compute_sigma(R));
}

PyObject* py_power_n(PyObject* self, PyObject* args)
{
  PyObject* py_ps;
  
  if(!PyArg_ParseTuple(args, "O", &py_ps))
     return NULL;

  PowerSpectrum* const ps=
    (PowerSpectrum *) PyCapsule_GetPointer(py_ps, "_PowerSpectrum");
  py_assert_ptr(ps);

  return Py_BuildValue("i", ps->n);
}

PyObject* py_power_ki(PyObject* self, PyObject* args)
{
  PyObject* py_ps;
  int i;
  
  if(!PyArg_ParseTuple(args, "Oi", &py_ps, &i))
     return NULL;

  PowerSpectrum* const ps=
    (PowerSpectrum *) PyCapsule_GetPointer(py_ps, "_PowerSpectrum");
  py_assert_ptr(ps);

  if(i < 0 || i >= ps->n) {
    PyErr_SetNone(PyExc_IndexError);
    return NULL;
  }
    
  return Py_BuildValue("d", exp(ps->log_k[i]));
}

PyObject* py_power_Pi(PyObject* self, PyObject* args)
{
  PyObject* py_ps;
  int i;
  
  if(!PyArg_ParseTuple(args, "Oi", &py_ps, &i))
     return NULL;

  PowerSpectrum* const ps=
    (PowerSpectrum *) PyCapsule_GetPointer(py_ps, "_PowerSpectrum");
  py_assert_ptr(ps);

  if(i < 0 || i >= ps->n) {
    PyErr_SetNone(PyExc_IndexError);
    return NULL;
  }

  return Py_BuildValue("d", exp(ps->log_P[i]));
}

PyObject* py_power_i(PyObject* self, PyObject* args)
{
  PyObject* py_ps;
  int i;
  
  if(!PyArg_ParseTuple(args, "Oi", &py_ps, &i))
     return NULL;

  PowerSpectrum* const ps=
    (PowerSpectrum *) PyCapsule_GetPointer(py_ps, "_PowerSpectrum");
  py_assert_ptr(ps);

  if(i < 0 || i >= ps->n) {
    PyErr_SetNone(PyExc_IndexError);
    return NULL;
  }

  return Py_BuildValue("dd", exp(ps->log_k[i]), exp(ps->log_P[i]));
}
