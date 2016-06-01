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
static Float boxsize;

static FFT* fft_pm;
static complex_t* delta_k;

static inline void grid_assign(Float * const d, 
	    const size_t ix, const size_t iy, const size_t iz, const Float f)
{
#ifdef _OPENMP
  #pragma omp atomic
#endif
  d[(ix*nc + iy)*ncz + iz] += f;
}

static inline Float grid_val(Float const * const d,
			const size_t ix, const size_t iy, const size_t iz)
{
  return d[(ix*nc + iy)*ncz + iz];
}


static void check_total_density(Float const * const density);
static void compute_delta_k();
static void compute_force_mesh(const int axis);
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
	     
  
  Float* const density= fft_pm->fx;
  const size_t local_nx= fft_pm->local_nx;
  const size_t local_ix0= fft_pm->local_ix0;
 
  msg_printf(msg_verbose, "particle position -> density mesh\n");

  const Float dx_inv= nc/boxsize;
  const Float fac= pm_factor*pm_factor*pm_factor;

#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(size_t i=0; i<np; i++) {
    Float x=p[i].x[0]*dx_inv;
    Float y=p[i].x[1]*dx_inv;
    Float z=p[i].x[2]*dx_inv;

#ifdef CHECK
    assert(0 <= x && x <= nc);
    assert(0 <= y && y <= nc);
    assert(0 <= z && z <= nc);
#endif
    
    int ix0= (int) x; // without floor, -1 < X < 0 is mapped to iI=0
    int iy0= (int) y; // assuming y,z are positive
    int iz0= (int) z;

    // CIC weight on left grid
    Float wx1= x - ix0;
    Float wy1= y - iy0;
    Float wz1= z - iz0;

    // CIC weight on right grid
    Float wx0= 1 - wx1;
    Float wy0= 1 - wy1;
    Float wz0= 1 - wz1;

    if(ix0 >= nc) ix0= 0;
    if(iy0 >= nc) iy0= 0; 
    if(iz0 >= nc) iz0= 0;

    int ix1= ix0 + 1; if(ix1 >= nc) ix1 -= nc;
    int iy1= iy0 + 1; if(iy1 >= nc) iy1 -= nc; // assumes y,z < boxsize
    int iz1= iz0 + 1; if(iz1 >= nc) iz1 -= nc;

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

template <class T>
void force_at_particle_locations(T const * const p, const int np, 
				 const int axis, Float3* const f)
{
  const Float dx_inv= nc/boxsize;
  const size_t local_nx= fft_pm->local_nx;
  const size_t local_ix0= fft_pm->local_ix0;
  const Float* fx= fft_pm->fx;

#ifdef _OPENMP
  #pragma omp parallel for default(shared)     
#endif
  for(size_t i=0; i<np; i++) {
    Float x= p[i].x[0]*dx_inv;
    Float y= p[i].x[1]*dx_inv;
    Float z= p[i].x[2]*dx_inv;

    int ix0= (int) x;
    int iy0= (int) y;
    int iz0= (int) z;
    
    Float wx1= x - ix0;
    Float wy1= y - iy0;
    Float wz1= z - iz0;

    Float wx0= 1 - wx1;
    Float wy0= 1 - wy1;
    Float wz0= 1 - wz1;

    if(ix0 >= nc) ix0= 0;
    if(iy0 >= nc) iy0= 0;
    if(iz0 >= nc) iz0= 0;

#ifdef CHECK
    assert(0 <= ix0 && ix0 < nc &&
	   0 <= iy0 && iy0 < nc &&
	   0 <= iz0 && iz0 < nc);
#endif

            
    int ix1= ix0 + 1; if(ix1 >= nc) ix1 -= nc;
    int iy1= iy0 + 1; if(iy1 >= nc) iy1 -= nc;
    int iz1= iz0 + 1; if(iz1 >= nc) iz1 -= nc;

#ifdef CHECK
    assert(0 <= ix1 && ix1 < nc &&
	   0 <= iy1 && iy1 < nc &&
	   0 <= iz1 && iz1 < nc);
#endif

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

//
// Public functions
//
void pm_init(const int nc_pm, const double pm_factor_,
	     Mem* const mem_pm, Mem* const mem_density,
	     const Float boxsize_)
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
  Float* const density= fft_pm->fx;
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
  pm_compute_density(particles);

#ifdef CHECK
  check_total_density(fft_pm->fx);
#endif

  msg_printf(msg_verbose, "PM force computation...\n");
  
  compute_delta_k();

  for(int axis=0; axis<3; axis++) {
    // delta(k) -> f(x_i)
    compute_force_mesh(axis);

    //msg_printf(msg_debug, "Force at particle location (local)\n");
    force_at_particle_locations<Particle>(
      particles->p, particles->np_local, axis, particles->force);

    //msg_printf(msg_debug, "Force at particle location (buffer)\n");
    force_at_particle_locations<Pos>(
      domain_buffer_positions(), domain_buffer_np(), axis,
      domain_buffer_forces());
  }

  // debug !!!
  /*
  Float3* ff= domain_buffer_forces();
  for(int i=0; i<domain_buffer_np(); ++i)
    printf("ff %e %e %e\n", ff[i][0], ff[i][1], ff[i][2]);
  */
  
  domain_get_forces(particles);
}

FFT* pm_compute_density(Particles* const particles)
{
  // Compute density only
  msg_printf(msg_verbose, "PM density computation...\n");

  domain_init(fft_pm, particles);
  domain_send_positions(particles);

  clear_density();
  pm_assign_cic_density<Particle>(particles->p, particles->np_local);
  pm_assign_cic_density<Pos>(domain_buffer_positions(), domain_buffer_np());

  return fft_pm;
}

//
// Static functions
//


void check_total_density(Float const * const density)
{
  // Checks <delta> = 0
  // Input: delta(x)
  double sum= 0.0;
  const size_t local_nx= fft_pm->local_nx;
  
  for(size_t ix=0; ix<local_nx; ix++)
    for(size_t iy=0; iy<nc; iy++)
      for(size_t iz=0; iz<nc; iz++)
	sum += density[(ix*nc + iy)*ncz + iz];

  double sum_global;
  MPI_Reduce(&sum, &sum_global, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  if(comm_this_node() == 0) {
    double tol= 10*FLOAT_EPS*nc*nc*nc;

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


void compute_delta_k()
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

  const Float f1= -1.0/pow(nc, 3.0)/(2.0*M_PI/boxsize);
  const size_t nckz=nc/2+1;
  const size_t local_nky= fft_pm->local_nky;
  const size_t local_iky0= fft_pm->local_iky0;


#ifdef _OPENMP
#pragma omp parallel for default(shared)
#endif
  for(size_t iy_local=0; iy_local<local_nky; iy_local++) {
    int iy= iy_local + local_iky0;
    int iy0= iy <= (nc/2) ? iy : iy - nc;

    Float k[3];
    k[1]= (Float) iy0;

    for(size_t ix=0; ix<nc; ix++) {
      int ix0= ix <= (nc/2) ? ix : ix - nc;
      k[0]= (Float) ix0;

      int kzmin= (ix==0 && iy==0); // skip (0,0,0) to avoid zero division

      for(size_t iz=kzmin; iz<nckz; iz++){
	k[2]= (Float) iz;

	Float f2= f1/(k[0]*k[0] + k[1]*k[1] + k[2]*k[2])*k[axis];

	size_t index= (nc*iy_local + ix)*nckz + iz;
	fk[index][0]= -f2*delta_k[index][1];
	fk[index][1]=  f2*delta_k[index][0];
      }
    }
  }

  fft_pm->mode= fft_mode_k;
  fft_pm->execute_inverse(); // f_k -> f(x)
}

// Does 3-linear interpolation
// particles= Values of mesh at particle positions P.x

/*
void add_buffer_forces(Particles* const particles, const size_t np)
{
  // !!! Non-MPI version
  Particle* const p= particles->p;
  
  const int np_local= particles->np_local;
  Float3* const force= particles->force;

  for(int j=np_local; j<np; j++) {
    size_t i= p[j].id - 1;
    assert(p[i].id == p[j].id);
    force[i][0] += force[j][0];
    force[i][1] += force[j][1];
    force[i][2] += force[j][2];
  }
}
*/

FFT* pm_get_fft()
{
  return fft_pm;
}
