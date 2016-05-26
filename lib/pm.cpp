#include <iostream>
#include <cstdio>
#include <cmath>
#include <cassert>
#include <gsl/gsl_rng.h>

#include "msg.h"
#include "mem.h"
#include "config.h"
#include "cosmology.h"
#include "comm.h"
#include "particle.h"
#include "fft.h"
#include "util.h"
#include "error.h"
#include "pm.h"
#include "pm_domain.h"

static double pm_factor;
static size_t nc, ncz;
static float_t boxsize;

static FFT* fft_pm;
static complex_t* delta_k;

static inline void grid_assign(float_t * const d, 
	    const size_t ix, const size_t iy, const size_t iz, const float_t f)
{
#ifdef _OPENMP
  #pragma omp atomic
#endif
  d[(ix*nc + iy)*ncz + iz] += f;
}

static inline float_t grid_val(float_t const * const d,
			const size_t ix, const size_t iy, const size_t iz)
{
  return d[(ix*nc + iy)*ncz + iz];
}


static void check_total_density(float_t const * const density);
static void compute_delta_k(void);
static void compute_force_mesh(const int k);
static void force_at_particle_locations(
		 Particles* const particles, const int np, const int axis);
static void add_buffer_forces(Particles* const particles, const size_t np);
static void clear_density();

//
// Template functions
//
template<class T>
void pm_assign_cic_density(T const * const p, size_t np) 
{
  // Assign CIC density to fft_pm using np particles P* p.x

  // Input:  particle positions in p[i].x for 0 <= i < np
  // Result: density field delta(x) in fft_pm->fx

  // particles are assumed to be periodiclly wraped up in y,z direction

  msg_printf(msg_verbose, "Computing PM density with %lu particles\n", np);
	     
  
  float_t* const density= (float*) fft_pm->fx;
  const size_t local_nx= fft_pm->local_nx;
  const size_t local_ix0= fft_pm->local_ix0;
 
  msg_printf(msg_verbose, "particle position -> density mesh\n");

  const float_t dx_inv= nc/boxsize;
  const float_t fac= pm_factor*pm_factor*pm_factor;

#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(size_t i=0; i<np; i++) {
    float x=p[i].x[0]*dx_inv;
    float y=p[i].x[1]*dx_inv;
    float z=p[i].x[2]*dx_inv;

    int ix0= (int) floor(x); // without floor, -1 < X < 0 is mapped to iI=0
    int iy0= (int) y;        // assuming y,z are positive
    int iz0= (int) z;

    // CIC weight on left grid
    float_t wx1= x - ix0;
    float_t wy1= y - iy0;
    float_t wz1= z - iz0;

    // CIC weight on right grid
    float_t wx0= 1 - wx1;
    float_t wy0= 1 - wy1;
    float_t wz0= 1 - wz1;

#ifdef CHECK
    assert(y >= 0.0f && z >= 0.0f);
#endif
            
    // No periodic wrapup in x direction. 
    // Buffer particles are copied from adjacent nodes, instead
    if(iy0 >= nc) iy0= 0; 
    if(iz0 >= nc) iz0= 0;

    int ix1= ix0 + 1;
    int iy1= iy0 + 1; if(iy1 >= nc) iy1= 0; // assumes y,z < boxsize
    int iz1= iz0 + 1; if(iz1 >= nc) iz1= 0;

    ix0 -= local_ix0;
    ix1 -= local_ix0;

    if(0 <= ix0 && ix0 < local_nx) {
      grid_assign(density, ix0, iy0, iz0, fac*wx0*wy0*wz0);
      grid_assign(density, ix0, iy0, iz1, fac*wx0*wy0*wz1);
      grid_assign(density, ix0, iy1, iz0, fac*wx0*wy1*wz0);
      grid_assign(density, ix0, iy1, iz1, fac*wx0*wy1*wz1);
    }

    if(0 <= ix1 && ix1 < local_nx) {
      grid_assign(density, ix1, iy0, iz0, fac*wx1*wy0*wz0);
      grid_assign(density, ix1, iy0, iz1, fac*wx1*wy0*wz1);
      grid_assign(density, ix1, iy1, iz0, fac*wx1*wy1*wz0);
      grid_assign(density, ix1, iy1, iz1, fac*wx1*wy1*wz1);
    }
  }

  fft_pm->mode= fft_mode_x;
  msg_printf(msg_verbose, "CIC density assignment finished.\n");
}

//
// Public functions
//
void pm_init(const int nc_pm, const double pm_factor_,
	     Mem* const mem_pm, Mem* const mem_density,
	     const float_t boxsize_,
	     const size_t np_alloc)
{
  msg_printf(msg_verbose, "PM module init\n");
  nc= nc_pm;
  pm_factor= pm_factor_;
  ncz= 2*(nc/2 + 1);
  boxsize= boxsize_;

  if(nc <= 1) {
    msg_printf(msg_fatal, "Error: nc_pm (= %d) must be larger than 1\n",
	       nc_pm);
    throw RuntimeError();
  }

  const size_t nckz= nc/2 + 1;
  
  mem_pm->use_from_zero(0);
  fft_pm= new FFT("PM", nc, mem_pm, 1);

  size_t size_density_k= nc*(fft_pm->local_nky)*nckz*sizeof(complex_t);
  delta_k= (complex_t*) mem_density->use_from_zero(size_density_k);

  
}

void clear_density()
{
  float_t* const density= (float*) fft_pm->fx;
  const size_t local_nx= fft_pm->local_nx;
    
#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(size_t ix = 0; ix < local_nx; ix++)
    for(size_t iy = 0; iy < nc; iy++)
      for(size_t iz = 0; iz < nc; iz++)
	density[(ix*nc + iy)*ncz + iz] = -1;
}

void pm_compute_force(Particles* const particles)
{
  // Main routine of this source file
  msg_printf(msg_verbose, "PM force computation...\n");

  ////size_t np_plus_buffer=
  ////send_buffer_positions(particles);
  ////pm_assign_cic_density(particles, np_plus_buffer);
  //pm_assign_cic_density(particles->p, particles->np_local);
  

#ifdef CHECK
  check_total_density(fft_pm->fx);
#endif
  
  compute_delta_k();

  for(int axis=0; axis<3; axis++) {
    // delta(k) -> f(x_i)
    compute_force_mesh(axis);

    // ToDo
    //force_at_particle_locations(particles, np_plus_buffer, axis);
  }
  ////add_buffer_forces(particles, np_plus_buffer);
}

FFT* pm_compute_density(Particles* const particles)
{
  // Compute density only
  msg_printf(msg_verbose, "PM density computation...\n");

  domain_init(fft_pm, particles);
  domain_send_positions(particles);

  clear_density();
  pm_assign_cic_density<Particle>(particles->p, particles->np_local);
  pm_assign_cic_density<Vec3>(domain_buffer_positions(), domain_buffer_np());

  return fft_pm;
}

//
// Static functions
//


void check_total_density(float_t const * const density)
{
  // Checks <delta> = 0
  // Input: delta(x)
  double sum= 0.0;
  const size_t local_nx= fft_pm->local_nx;
  
  for(size_t ix = 0; ix < local_nx; ix++)
    for(size_t iy = 0; iy < nc; iy++)
      for(size_t iz = 0; iz < nc; iz++)
	sum += density[(ix*nc + iy)*ncz + iz];

  double sum_global;
  MPI_Reduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(comm_this_node() == 0) {
    double tol= FLOAT_EPS*nc*nc*nc;

    if(fabs(sum_global) > tol) {
      msg_printf(msg_error,
		 "Error: total CIC density error is  too large: %le > %le\n", 
		 sum_global, tol);
      throw AssertionError();
    }

    msg_printf(msg_debug, 
	      "Total CIC density OK within machine precision: %lf (< %.2lf).\n",
	       sum_global, tol);
  }
}


void compute_delta_k(void)
{
  // Fourier transform delta(x) -> delta(k) and copy it to delta_k
  //  Input:  delta(x) in fft_pm->fx
  //  Output: delta(k) in delta_k

  msg_printf(msg_verbose, "delta(x) -> delta(k)\n");
  fft_pm->execute_forward();

  // Copy density(k) in fft_pm to density_k
  const size_t nckz= nc/2 + 1;
  const size_t local_nky= fft_pm->local_nky;

  complex_t* pm_k= fft_pm->fk;
  
#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(size_t iy=0; iy<local_nky; iy++) {
    for(size_t ix=0; ix<nc; ix++) {
      for(size_t iz=0; iz<nckz; iz++){
	size_t index= (nc*iy + ix)*nckz + iz;
	delta_k[index][0]= pm_k[index][0];
	delta_k[index][1]= pm_k[index][1];	
      }
    }
  }
}

void compute_force_mesh(const int axis)
{
  // Calculate one component of force mesh from precalculated density(k)
  //   Input:   delta(k)   mesh delta_k
  //   Output:  force_i(k) mesh fft_pm->fx

  complex_t* const fk= fft_pm->fk;
  
  //k=0 zero mode force is zero
  fk[0][0]= 0;
  fk[0][1]= 0;

  const float_t f1= -1.0/pow(nc, 3.0)/(2.0*M_PI/boxsize);
  const size_t nckz=nc/2+1;
  const size_t local_nky= fft_pm->local_nky;
  const size_t local_iky0= fft_pm->local_iky0;


#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
  for(size_t iy_local=0; iy_local<local_nky; iy_local++) {
    int iy= iy_local + local_iky0;
    int iy0= iy <= (nc/2) ? iy : iy - nc;

    float_t k[3];
    k[1]= (float_t) iy0;

    for(size_t ix=0; ix<nc; ix++) {
      int ix0= ix <= (nc/2) ? ix : ix - nc;
      k[0]= (float_t) ix0;

      int kzmin= (ix==0 && iy==0); // skip (0,0,0) to avoid zero division

      for(size_t iz=kzmin; iz<nckz; iz++){
	k[2]= (float_t) iz;

	float f2= f1/(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])*k[axis];

	size_t index= (nc*iy_local + ix)*nckz + iz;
	fk[index][0]= -f2*delta_k[index][1];
	fk[index][1]=  f2*delta_k[index][0];
      }
    }
  }

  fft_pm->execute_inverse(); // f_k -> f(x)
}

// Does 3-linear interpolation
// particles= Values of mesh at particle positions P.x
void force_at_particle_locations(Particles* const particles, const int np, 
				 const int axis)
{
  const Particle* p= particles->p;
  
  const float_t dx_inv= nc/boxsize;
  const size_t local_nx= fft_pm->local_nx;
  const size_t local_ix0= fft_pm->local_ix0;
  const float_t* fx= fft_pm->fx;
  float3* f= particles->force;
  
#ifdef _OPENMP
  #pragma omp parallel for default(shared)     
#endif
  for(size_t i=0; i<np; i++) {
    float_t x=p[i].x[0]*dx_inv;
    float_t y=p[i].x[1]*dx_inv;
    float_t z=p[i].x[2]*dx_inv;
            
    int ix0= (int) floor(x);
    int iy0= (int) y;
    int iz0= (int) z;
    
    float_t wx1= x - ix0;
    float_t wy1= y - iy0;
    float_t wz1= z - iz0;

    float_t wx0= 1 - wx1;
    float_t wy0= 1 - wy1;
    float_t wz0= 1 - wz1;

    if(iy0 >= nc) iy0= 0;
    if(iz0 >= nc) iz0= 0;
            
    int ix1= ix0 + 1;
    int iy1= iy0 + 1; if(iy1 >= nc) iy1= 0;
    int iz1= iz0 + 1; if(iz1 >= nc) iz1= 0;

    ix0 -= local_ix0;
    ix1 -= local_ix0;

    f[i][axis]= 0;

    if(0 <= ix0 && ix0 < local_nx) {
      f[i][axis] += 
	grid_val(fx, ix0, iy0, iz0)*wx0*wy0*wz0 +
	grid_val(fx, ix0, iy0, iz1)*wx0*wy0*wz1 +
	grid_val(fx, ix0, iy1, iz0)*wx0*wy1*wz0 +
	grid_val(fx, ix0, iy1, iz1)*wx0*wy1*wz1;
    }
    if(0 <= ix1 && ix1 < local_nx) {
      f[i][axis] += 
	grid_val(fx, ix1, iy0, iz0)*wx1*wy0*wz0 +
	grid_val(fx, ix1, iy0, iz1)*wx1*wy0*wz1 +
	grid_val(fx, ix1, iy1, iz0)*wx1*wy1*wz0 +
	grid_val(fx, ix1, iy1, iz1)*wx1*wy1*wz1;
    }
  }
}

void add_buffer_forces(Particles* const particles, const size_t np)
{
  // !!! Non-MPI version
  Particle* const p= particles->p;
  
  const int np_local= particles->np_local;
  float3* const force= particles->force;

  for(int j=np_local; j<np; j++) {
    size_t i= p[j].id - 1;
    assert(p[i].id == p[j].id);
    force[i][0] += force[j][0];
    force[i][1] += force[j][1];
    force[i][2] += force[j][2];
  }
}

FFT* pm_get_fft()
{
  return fft_pm;
}
