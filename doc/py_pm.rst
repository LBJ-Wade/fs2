PM
====

Particle Mesh force computation

.. py:function::
   pm_init(nc, pm_factor, boxsize)

   Initialise the pm module.

   :param int nc_pm: number of mesh per dimension
   :param int pm_factor: nc_pm/nc
   :param float boxsize: length in :math:`[h^{-1} \mathrm{Mpc}]`
   
.. py:function::
   pm_compute_force(particles)

   Compute the density on a mesh and force at particle positions.

   :param Particles particles:
   
.. py:function::
   pm_compute_density(particles)

   Compute the density only.

   :param Particles particles:
   :rtype: :py:class:`FFT`


