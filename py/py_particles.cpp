#include <iostream>
#include <vector>
#include "msg.h"
#include "particle.h"
#include "util.h"
#include "py_particles.h"
#include "py_assert.h"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"

using namespace std;

static vector<Particle>* v= 0;

PyMODINIT_FUNC
py_particles_module_init()
{
  import_array();

  return NULL;
}

PyObject* py_particles_alloc(PyObject* self, PyObject* args)
{
  int nc;
  double boxsize;
  
  if(!PyArg_ParseTuple(args, "id", &nc, &boxsize)) {
    return NULL;
  }

  Particles* const particles = new Particles(nc, boxsize);

  return PyCapsule_New(particles, "_Particles", py_particles_free);
}

void py_particles_free(PyObject *obj)
{
  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(obj, "_Particles");
  py_assert_void(particles);

  delete particles;
}

PyObject* py_particles_len(PyObject* self, PyObject* args)
{
  PyObject* py_particles;
  
  if(!PyArg_ParseTuple(args, "O", &py_particles))
     return NULL;

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  return Py_BuildValue("k", (unsigned long) particles->np_local);
}

PyObject* py_particles_slice(PyObject* self, PyObject* args)
{
  // _particles_slice(_particles, frac)
  // Return particles 0 < x[2] < frac*boxsize as an np.array
  PyObject* py_particles;
  double frac;
  
  if(!PyArg_ParseTuple(args, "Od", &py_particles, &frac))
     return NULL;

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);

  const float_t boxsize= particles->boxsize;
  const float_t x_max= frac*boxsize;
  Particle* const p= particles->p;
  const size_t n= particles->np_local;

  if(v == 0) v= new vector<Particle>();
  v->clear();

  for(size_t i=0; i<n; ++i) {
    Particle pp = p[i];
    periodic_wrapup_p(pp, boxsize);
    if(pp.x[2] < x_max)   
      v->push_back(pp);
  }

  // Return vector<Particle> as np.array
  const int nd=2;
  const int ncol= sizeof(Particle)/sizeof(float_t);
  npy_intp dims[]= {(npy_intp) v->size(), ncol};

  return PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, &(v->front()));
}

PyObject* py_particles_getitem(PyObject* self, PyObject* args)
{
  if(v == 0) v= new vector<Particle>();
  v->clear();
  
  PyObject *py_particles, *py_index;
  
  if(!PyArg_ParseTuple(args, "OO", &py_particles, &py_index))
    return NULL;

  Particles* const particles=
    (Particles *) PyCapsule_GetPointer(py_particles, "_Particles");
  py_assert_ptr(particles);
  Particle* const p= particles->p;
  const size_t n= particles->np_local;

  if(PyNumber_Check(py_index)) {
    // particles[i]; return ith particle
    long i= PyLong_AsLong(py_index);
    if(i < 0)
      i = n + i;

    if(i < 0 || i >= (long) particles->np_local) {
      PyErr_SetNone(PyExc_IndexError);
      return NULL;
    }
    
    return Py_BuildValue("(ddd)",
		 (double) p[i].x[0], (double) p[i].x[1], (double) p[i].x[2]);
  }
  else if(PySlice_Check(py_index)) {
    // particles[i:j:k]; return i to j with step k
    cout << "Slice object given\n";
    Py_ssize_t start, stop, step, length;
    const int ret=
      PySlice_GetIndicesEx(py_index, n, &start, &stop, &step, &length);
    if(ret)
      return NULL;

    for(int i=start; i<stop; i+=step) {
      if(0 <= i && i < (int) particles->np_local)
	v->push_back(p[i]);
    }
  }
  else {
    return NULL;
  }

  const int nd=2;
  const int ncol= sizeof(Particle)/sizeof(float_t);
  npy_intp dims[]= {(npy_intp) v->size(), ncol};

  return PyArray_SimpleNewFromData(nd, dims, NPY_FLOAT, &(v->front()));
}
