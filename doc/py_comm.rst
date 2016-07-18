####
comm
####

.. py:function:: comm_n_nodes()


   Return the number of MPI nodes
   
   :rtype: int

.. py:function:: comm_this_nodes()

   Return the index (rank) of this MPI node

   :rtype: int

.. py:function:: comm_mpi_finalise()

   Finalise the MPI programme. Must be called at the end of the script.
