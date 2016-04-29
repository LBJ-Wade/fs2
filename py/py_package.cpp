//
// Information for Python
//
#include <iostream>
#include "Python.h"

#include "py_msg.h"

using namespace std;

static PyMethodDef methods[] = {
  {"set_loglevel", py_msg_set_loglevel, METH_VARARGS,
   "set loglevel: 0=debug, 1=verbose, ..."},
  
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
PyInit__mockgallib(void) {
  //py_power_module_init();  
  
  return PyModule_Create(&module);
}
