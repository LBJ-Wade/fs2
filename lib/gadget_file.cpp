//
// Writes Gadget binary file
//
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cassert>
#include "msg.h"
#include "comm.h"
#include "error.h"
#include "cosmology.h"
#include "gadget_file.h"

void gadget_file_write_particles(const char filebase[],
				 Particles const * const particles,
				 int use_long_id, const double h)
{
  char filename[256];
  if(comm_n_nodes() == 1) {
    sprintf(filename, "%s", filebase);
  }
  else {
    sprintf(filename, "%s.%d", filebase, comm_this_node());
  }

  FILE* fp= fopen(filename, "w");
  if(fp == 0) {
    msg_printf(msg_error, "Error: Unable to write to file: %s\n", filename);
    throw IOError();
  }

  Particle* const p= particles->p;
  const int np= particles->np_local;
  const double boxsize= particles->boxsize;
  const double omega_m= cosmology_omega_m();

  if(use_long_id)
    msg_printf(msg_info, "Longid is used for GADGET particles. %d-byte.\n", 
	       sizeof(unsigned long long));
  else
    msg_printf(msg_info, "ID is %d-byte unsigned int\n", sizeof(unsigned int));

  long long np_total= comm_sum<long long>(np);

  GadgetHeader header; assert(sizeof(GadgetHeader) == 256);
  memset(&header, 0, sizeof(GadgetHeader));

  const double m= cosmology_rho_m()*pow(boxsize, 3.0)/np_total;
  
  header.np[1]= np;
  header.mass[1]= m;
  header.time= particles->a_x;
  header.redshift= 1.0/header.time - 1;
  header.np_total[1]= (unsigned int) np_total;
  header.np_total_highword[1]= (unsigned int) (np_total >> 32);
  header.num_files= comm_n_nodes();
  header.boxsize= boxsize;
  header.omega0= omega_m;
  header.omega_lambda= 1.0 - omega_m;
  header.hubble_param= h;


  int blklen= sizeof(GadgetHeader);
  fwrite(&blklen, sizeof(blklen), 1, fp);
  fwrite(&header, sizeof(GadgetHeader), 1, fp);
  fwrite(&blklen, sizeof(blklen), 1, fp);

  // position
  blklen= np*sizeof(float)*3;
  fwrite(&blklen, sizeof(blklen), 1, fp);
  for(int i=0; i<np; i++)
    fwrite(p[i].x, sizeof(float), 3, fp);
  fwrite(&blklen, sizeof(blklen), 1, fp);

  // velocity
  const float vfac= 1.0/sqrt(particles->a_x); // Gadget convention

  fwrite(&blklen, sizeof(blklen), 1, fp);
  for(int i=0; i<np; i++) {
    float vout[]= {float(vfac*p[i].v[0]),
		   float(vfac*p[i].v[1]),
		   float(vfac*p[i].v[2])};
    fwrite(vout, sizeof(float), 3, fp);
  }
  fwrite(&blklen, sizeof(blklen), 1, fp);

  // id
  if(use_long_id) {
    blklen= np*sizeof(unsigned long long);
    fwrite(&blklen, sizeof(blklen), 1, fp);
    for(int i=0; i<np; i++) {
      unsigned long long id_out= p[i].id;
      fwrite(&id_out, sizeof(unsigned long long), 1, fp); 
    }
  }
  else {
    blklen= np*sizeof(unsigned int);
    fwrite(&blklen, sizeof(blklen), 1, fp);
    for(int i=0; i<np; i++) {
      unsigned int id_out= p[i].id;
      fwrite(&id_out, sizeof(unsigned int), 1, fp); 
    }
  }
  fwrite(&blklen, sizeof(blklen), 1, fp);


  
  fclose(fp);  

  msg_printf(msg_info, "particles %s written\n", filebase);
}

