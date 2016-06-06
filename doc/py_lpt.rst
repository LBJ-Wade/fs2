####
LPT
####

Module for Lagrangian Perturbation Theory (LPT)

.. py:function:: lpt(nc, boxsize, a, ps, seed)   
   
   Generate a random Gaussian initial condition and set 2LPT displacements.

   :param int nc: number of particles per dimension
   :param float boxsize: size of periodic box size on a side :math:`[h^{-1} \mathrm{Mpc}]`
   :param float a: scale factor for the particle positions
   :param PowerSpectrum ps: input linear power spectrum
   :param long seed: random seed for the initial Gaussian random field
   :rtype: :py:class:`Particles`

Example
=======
   
.. code-block:: python

   import fs

   # cosmological parameters
   omega_m = 0.308
   nc = 64           # nc**3 particles
   boxsize = 64
   a = 0.1           # scale factor
   seed = 1          # random seed for the random Gaussian field

   # initial setup
   fs.set_loglevel(1)
   fs.cosmology_init(omega_m)
   ps = fs.PowerSpectrum('data/planck_matterpower.dat')

   # Set 2LPT displacements at scale factor a
   particles = fs.lpt(nc, boxsize, a, ps, seed)

   fs.comm_mpi_finalise()


