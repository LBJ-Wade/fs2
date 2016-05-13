#ifndef COMM_H
#define COMM_H 1

#include <mpi.h>

void comm_mpi_init(int* pargc, char*** pargv);
void comm_mpi_finalise();
void comm_abort();
int  comm_this_node();
int  comm_n_nodes();
void comm_bcast_int(int* p_int, int count);
void comm_bcast_double(double* p_double, int count);

template<class T> void comm_bcast(T x, MPI_Datatype datatype)
{
  MPI_Bcast(&x, 1, datatype, 0, MPI_COMM_WORLD);
}

template<class T> T comm_sum(T x, MPI_Datatype datatype)
{
  T x_reduced;
  MPI_Allreduce(&x, &x_reduced, 1, datatype, MPI_SUM, MPI_COMM_WORLD);

  return x_reduced;
}


#endif
