#include <stdio.h>
#include <assert.h>

#include "comm.h"
#include "msg.h"

//Particles* alloc_particles(const int nc);

int main(int argc, char* argv[])
{
  // Setup MPI Init  as comm_mpi_init?
  //
  comm_mpi_init(&argc, &argv);
  msg_set_loglevel(msg_debug);

  msg_printf(msg_debug, "Hello world\n");
  
  //PowerSpectrum* ps= power_alloc("camb_matterpower.dat", 0.812);
  

  comm_mpi_finalise();
  return 0;
}

