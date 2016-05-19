//
// Information for Python
//
#include <iostream>
#include "Python.h"

#include "py_msg.h"
#include "py_comm.h"
#include "py_cosmology.h"
#include "py_power.h"
#include "py_particles.h"
#include "py_lpt.h"
#include "py_pm.h"
#include "py_cola.h"
#include "py_leapfrog.h"
#include "py_write.h"
#include "py_hdf5_io.h"
#include "py_fft.h"

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
  {"comm_this_node", py_comm_this_node, METH_VARARGS,
   "return index (rank) of this node"},
  {"comm_n_nodes", py_comm_n_nodes, METH_VARARGS,
   "return number of MPI nodes (size)"}, 

  {"_cosmology_init", py_cosmology_init, METH_VARARGS,
   "cosmology_init(omega_m0, h=0.7); set omega_m and h"},   
  
  {"_power_alloc", py_power_alloc, METH_VARARGS,
   "allocate a new _ps opbject"},
  {"_power_n", py_power_alloc, METH_VARARGS,
   "_power_n(_ps); get number of P(k) data"},
  {"_power_i", py_power_i, METH_VARARGS,
   "_power_i(_ps, i); get (k[i], P[i])"},

  {"_particles_alloc", py_particles_alloc, METH_VARARGS,
   "allocate a new _particles object"},
  {"_particles_slice", py_particles_slice, METH_VARARGS,
   "return a slice of particles as np.array"},
  {"_particles_len", py_particles_len, METH_VARARGS,
   "return the number particles"},
  {"_particles_getitem", py_particles_getitem, METH_VARARGS,
   "_particles_getitem(_particles, row, col)"},
  
  {"_lpt", py_lpt, METH_VARARGS,
   "_lpt(nc, boxsize, a, _ps, rando_seed); setup 2LPT displacements"},

  {"pm_init", py_pm_init, METH_VARARGS,
   "_pm_init(nc_pm, pm_factor, boxsize); initialise pm module"},
  {"_pm_compute_force", py_pm_compute_force, METH_VARARGS,
   "_pm_compute_force(_particles)"},   
  
  {"_cola_kick", py_cola_kick, METH_VARARGS,
   "_cola_kick(_particles, a_vel); update particle velocities to a_vel"},
  {"_cola_drift", py_cola_drift, METH_VARARGS,
   "_cola_drift(_particles, a_pos); update particle positions to a_pos"},

  {"_leapfrog_initial_velocity", py_leapfrog_initial_velocity, METH_VARARGS,
   "_leapfrog_initial_velocity(_particles, a_pos"},
  {"_leapfrog_kick", py_leapfrog_kick, METH_VARARGS,
   "_leapfrog_kick(_particles, a_vel); update particle velocities to a_vel"},
  {"_leapfrog_drift", py_leapfrog_drift, METH_VARARGS,
   "_leapfrog_drift(_particles, a_pos); update particle positions to a_pos"},

  {"_write_gadget_binary", py_write_gadget_binary, METH_VARARGS,
   "_write_gadget_binary(_particles, filename, use_long_id"},   

  {"_hdf5_write_particles", py_hdf5_write_particles, METH_VARARGS,
   "_hdf5_write_particles(_particles, filename)"},
  
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
  py_particles_module_init();
  py_fft_module_init();
  
  return PyModule_Create(&module);
}
