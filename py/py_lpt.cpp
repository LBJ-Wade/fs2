#include "particle.h"
#include "lpt.h"
#include "py_assert.h"
#include "py_particles.h"
#include "py_lpt.h"

PyObject* py_lpt(PyObject* self, PyObject* args)
{
  //_lpt(nc, boxsize, a, _ps, seed

  PyObject* py_ps;
  int nc;
  double a, boxsize;
  unsigned long seed;


  if(!PyArg_ParseTuple(args, "iddkO", &nc, &boxsize, &a,  &seed, &py_ps)) {
    return NULL;
  }

  PowerSpectrum* const ps=
    (PowerSpectrum *) PyCapsule_GetPointer(py_ps, "_PowerSpectrum");
  py_assert_ptr(ps);

  Particles* particles= new Particles(nc, boxsize);
    
  size_t mem_size= 9*fft_mem_size(nc, 0);
  Mem* const mem= new Mem("LPT", mem_size);

  lpt_init(nc, boxsize, mem);
  lpt_set_displacements(seed, ps, a, particles);


  delete mem;
  
  return PyCapsule_New(particles, "_Particles", py_particles_free);  
}

PyObject* py_lpt_set_offset(PyObject* self, PyObject* args)
{
  double offset;
  
  if(!PyArg_ParseTuple(args, "d", &offset)) {
    return NULL;
  }
  
  lpt_set_offset(offset);

  Py_RETURN_NONE;
}
