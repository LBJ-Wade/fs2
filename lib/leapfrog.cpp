#include <math.h>
#include <assert.h>
#include <mpi.h>

#include <gsl/gsl_integration.h>
#include <gsl/gsl_roots.h>
#include <gsl/gsl_sf_hyperg.h> 
#include <gsl/gsl_errno.h>

#include "particle.h"
#include "msg.h"
#include "leapfrog.h"
#include "cosmology.h"

static double SqStd(double ai, double af);
static double SphiStd(double ai, double af);

void leapfrog_set_initial_velocity(Particles* const particles, const double a)
{
  Particle* const p= particles->p;
  const int np= particles->np_local;

  const Float da1= cosmology_D_growth(a);
  const Float da2= cosmology_D2_growth(a, da1);
    
  const Float Dv= cosmology_Dv_growth(a, da1);
  const Float D2v= cosmology_D2v_growth(a, da2);

  for(int i=0; i<np; i++) {
    p[i].v[0]= p[i].dx1[0]*Dv + p[i].dx2[0]*D2v;
    p[i].v[1]= p[i].dx1[1]*Dv + p[i].dx2[1]*D2v;
    p[i].v[2]= p[i].dx1[2]*Dv + p[i].dx2[2]*D2v;
  }

  particles->a_v= a;
  
  msg_printf(msg_info, "Leapfrog (non-cola) initial velocity set at a= %.3f\n", a);
  msg_printf(msg_debug, "Dv= %e, Dv2= %e\n", Dv, D2v);
}

void leapfrog_kick(Particles* const particles, const double avel1)
{
  const double ai=  particles->a_v;  // t - 0.5*dt
  const double af=  avel1;           // t + 0.5*dt

  const double om= cosmology_omega_m();
  const Float kick_factor= SphiStd(ai, af);

  msg_printf(msg_info, "Leapfrog kick %lg -> %lg\n", ai, avel1);
  msg_printf(msg_debug, "kick_factor = %lg\n", kick_factor);

  Particle* const p= particles->p;
  const int np= particles->np_local;
  Float3* const f= particles->force;

  // Kick using acceleration at scale factor a
  // Assume forces at a is in particles->force
#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(size_t i=0; i<np; i++) {
    p[i].v[0] += -1.5*om*f[i][0]*kick_factor;
    p[i].v[1] += -1.5*om*f[i][1]*kick_factor;
    p[i].v[2] += -1.5*om*f[i][2]*kick_factor;
  }
  
  particles->a_v= avel1;
}

void leapfrog_drift(Particles* const particles, const double apos1)
{
  const double ai= particles->a_x;
  const double af= apos1;
  
  Particle* const p= particles->p;
  const size_t np= particles->np_local;

  const double dt=SqStd(ai, af);

  msg_printf(msg_info, "Leapfrog drift %lg -> %lg\n", ai, af);
  msg_printf(msg_debug, "dt = %lg\n", dt);

  // Drift
#ifdef _OPENMP
  #pragma omp parallel for default(shared)
#endif
  for(int i=0; i<np; i++) {
    p[i].x[0] += p[i].v[0]*dt;
    p[i].x[1] += p[i].v[1]*dt;
    p[i].x[2] += p[i].v[2]*dt;
  }
    
  particles->a_x= af;
}

static double funSphiStd (double a, void* params)
{
  const double om= *(double*) params;
  
  return 1.0/(sqrt(om/(a*a*a) + 1.0 - om)*a*a);
}

static double SphiStd(double ai, double af)
{
  gsl_integration_workspace* const w 
    = gsl_integration_workspace_alloc(5000);

  double omega_m= cosmology_omega_m();
     
  gsl_function F;
  F.function= &funSphiStd;
  F.params= &omega_m;

  double result, error;
  gsl_integration_qag(&F, ai, af, 0, 1e-5, 5000, 6,
		      w, &result, &error); 
  
  gsl_integration_workspace_free(w);
     
  return result;
}

static double funSqStd (double a, void* params)
{
  const double om= *(double*) params;
  return 1.0/(sqrt(om/(a*a*a) + 1.0 - om)*a*a*a);
}
     
static double SqStd(double ai, double af)
{
  gsl_integration_workspace* const w 
    = gsl_integration_workspace_alloc(5000);
       
  double result, error;

  double omega_m= cosmology_omega_m();
     
  gsl_function F;
  F.function= &funSqStd;
  F.params= &omega_m;
     
  gsl_integration_qag(&F, ai, af, 0, 1e-5, 5000, 6, w, &result, &error);      
  gsl_integration_workspace_free(w);
  
  return result;
}
