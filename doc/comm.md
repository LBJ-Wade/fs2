comm
====

Interface for MPI communication.

void **comm_mpi_init**(&*argc*, &*argv*)

> Needs to be called first

void **comm_mpi_finalise**()

> Needs to be called at the end

void **comm_abort**()

> Abort for MPI

int **comm_n_nodes**()

> number of MPI nodes

int **comm_this_node**()

> index of this MPI node
