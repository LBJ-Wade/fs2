//
// Information for Python
//
#include <iostream>
#include "Python.h"

#include "py_msg.h"
#include "py_comm.h"
#include "py_mem.h"
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
#include "py_config.h"
#include "py_timer.h"
#include "py_stat.h"

using namespace std;

static PyMethodDef methods[] = {
  {"set_loglevel", py_msg_set_loglevel, METH_VARARGS,
   "set loglevel: 0=debug, 1=verbose, ..."},

  {"comm_mpi_init", py_comm_mpi_init, METH_VARARGS,
   "Initialize MPI; called automatically by __init__.py"},
  {"comm_mpi_finalise", py_comm_mpi_finalise, METH_VARARGS,
   "Finalise MPI"},
  {"comm_hello", py_comm_hello, METH_VARARGS,
   "Test print statiment with MPI"},
  {"comm_this_node", py_comm_this_node, METH_VARARGS,
   "Return the index (rank) of this node"},
  {"comm_n_nodes", py_comm_n_nodes, METH_VARARGS,
   "Return the number of MPI nodes (MPI size)"}, 

  {"_mem_alloc", py_mem_alloc, METH_VARARGS,
   "create a new Mem object"},
  {"_mem", py_mem, METH_VARARGS,
   "return mem allocated and using"},
  
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
   "return the local number of particles"},
  {"_particles_np_total", py_particles_np_total, METH_VARARGS,
   "return the total number of particles"},
  {"_particles_getitem", py_particles_getitem, METH_VARARGS,
   "_particles_getitem(_particles, row, col)"},
  {"_particles_one", py_particles_one, METH_VARARGS,
   "_particles_one(_particles, x, y, z)"},
  {"_particles_update_np_total", py_particles_update_np_total, METH_VARARGS,
   "_particles_update_np_total(_particles)"},
  {"_particles_id_asarray", py_particles_id_asarray, METH_VARARGS,
   "_particles_id_asarray(_particles)"},
  {"_particles_x_asarray", py_particles_x_asarray, METH_VARARGS,
   "_particles_x_asarray(_particles)"},
  {"_particles_force_asarray", py_particles_force_asarray, METH_VARARGS,
   "_particles_force_asarray(_particles)"},
  
  {"_lpt", py_lpt, METH_VARARGS,
   "_lpt(nc, boxsize, a, _ps, rando_seed); setup 2LPT displacements"},
  {"_lpt_set_offset", py_lpt_set_offset, METH_VARARGS,
   "_lpt_set_offset(offset)"},

  {"_pm_init", py_pm_init, METH_VARARGS,
   "_pm_init(nc_pm, pm_factor, boxsize); initialise pm module"},
  {"_pm_compute_force", py_pm_compute_force, METH_VARARGS,
   "_pm_compute_force(_particles)"},   
  {"_pm_compute_density", py_pm_compute_density, METH_VARARGS,
   "_pm_compute_density(_particles); returns density mesh as np.array."},
  {"_pm_domain_init", py_pm_domain_init, METH_VARARGS,
   "_pm_domain_init(_particles)"},
  {"_pm_send_positions", py_pm_send_positions, METH_VARARGS,
   "_pm_send_positions(_particles)"},
  {"_pm_check_total_density", py_pm_check_total_density, METH_VARARGS,
   "_pm_check_total_density"},
  {"_pm_get_forces", py_pm_get_forces, METH_VARARGS,
   "_pm_get_forces(_particles)"},

  
  {"_pm_write_packet_info", py_pm_write_packet_info, METH_VARARGS,
   "_pm_write_packet_info(filename)"},
  {"_pm_set_packet_size", py_pm_set_packet_size, METH_VARARGS,
   "_pm_set_packet_size(packet_size)"},
  
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

  {"_fft_alloc", py_fft_alloc, METH_VARARGS,
   "_fft_alloc(nc)"},
  {"_fft_set_test_data", py_fft_set_test_data, METH_VARARGS,
   "_fft_set_test_data(_fft)"},
  {"_fft_fx_global_as_array", py_fft_fx_global_as_array, METH_VARARGS,
   "_fft_fx_global_as_array(_fft); return fft->fx as nc^3 np.array"},

  {"config_precision", py_config_precision, METH_VARARGS,
   "get 'single' or 'double'"},

  {"timer_save", py_timer_save, METH_VARARGS,
   "timer_save(filename)"},

  {"_stat_set_filename", py_stat_set_filename, METH_VARARGS,
   "_stat_set_filename(filename)"},

  {"_stat_record_pm_nbuf", py_stat_record_pm_nbuf, METH_VARARGS,
   "_stat_record_pm_nbuf(group_name)"},

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
  py_particles_module_init();
  py_fft_module_init();
  
  return PyModule_Create(&module);
}
