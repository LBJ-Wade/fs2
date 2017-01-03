m####
COLA
####

Module for COmoving Lagrangian Acceleration (COLA) time integration

.. py:function:: cola.drift(particles, a_pos)


   Evolve particle positions from `particles.a_pos` to `a_pos`
   
   :param Particles particles:
   :param float a_pos: scale factor after the drift	     


.. py:function:: cola.kick(particles, a_vel)

   Evolve particle positions from `particles.a_vel` to `a_vel`
   using `particles.force`

   :param Particles particles:
   :param float a_vel: scale factor after the drift
   

Example
=======
   
.. code-block:: python

   import fs

   # parameters
   omega_m = 0.308
   nc = 64
   nc_pm = nc        # number of Particle Mesh per dimension
   boxsize = 64
   a_init = 0.1      # initial and final scale factors
   a_final = 1.0
   seed = 1
   nsteps = 9

   fs.cosmology.init(omega_m)
   ps = fs.PowerSpectrum('data/planck_matterpower.dat')

   # random Gaussian initial condition
   particles = fs.lpt.init(nc, boxsize, a_init, ps, seed)

   fs.pm.init(nc_pm, nc_pm/nc, boxsize)

   # Evolve velocities and positions
   for i in range(nsteps):
       a_vel = a_init + (a_final - a_init)/nstep*(i + 0.5)
       fs.pm_compute_force(particles)
       fs.cola.kick(particles, a_vel)

       a_pos = a_init + (a_final - a_init)/nstep*(i + 1.0)
       fs.cola.drift(particles, a_pos)

   fs.comm_mpi_finalise()

