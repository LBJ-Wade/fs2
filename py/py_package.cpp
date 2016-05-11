//
// Information for Python
//
#include <iostream>
#include "Python.h"

#include "py_msg.h"
#include "py_comm.h"
#include "py_power.h"

using namespace std;

static PyMethodDef methods[] = {
  {"set_loglevel", py_msg_set_loglevel, METH_VARARGS,
   "set loglevel: 0=debug, 1=verbose, ..."},
  {"comm_mpi_init", py_comm_mpi_init, METH_VARARGS,
   "initialize MPI"},
  {"comm_mpi_finalise", py_comm_mpi_finalise, METH_VARARGS,
   "finalise MPI"},
  {"comm_hello", py_comm_hello, METH_VARARGS,
   "test print statiment with MPI"},

  {"_power_alloc", py_power_alloc, METH_VARARGS,
   "allocate a new _ps opbject"},
  {"_power_n", py_power_alloc, METH_VARARGS,
   "_power_n(_ps); get number of P(k) data"},
  {"_power_i", py_power_i, METH_VARARGS,
   "_power_i(_ps, i); get (k[i], P[i])"},
  
  {NULL, NULL, 0, NULL}
};


static struct PyModuleDef module = {
  PyModuleDef_HEAD_INIT,
  "_fs", // name of this module
  "A package for fast simulation", // Doc String
  -1,
  methods
};

PyMODINIT_FUNC
PyInit__fs(void) {
  //py_power_module_init();  
  
  return PyModule_Create(&module);
}