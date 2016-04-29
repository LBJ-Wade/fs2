#include "comm.h"
#include "py_comm.h"

PyObject* py_comm_mpi_init(PyObject *self, PyObject* args)
{
  comm_mpi_init(0, 0);
  Py_RETURN_NONE;
}

PyObject* py_comm_mpi_finalise(PyObject *self, PyObject* args)
{
  comm_mpi_finalise();
  Py_RETURN_NONE;
}

PyObject* py_comm_hello(PyObject *self, PyObject* args)
{
  printf("Hello from node %d / %d\n", comm_this_node(), comm_n_nodes());

  Py_RETURN_NONE;
}





