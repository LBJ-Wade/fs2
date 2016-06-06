import fs


omega_m = 0.308
nc = 64
nc_pm = nc
boxsize = 64
a_init = 0.1
a_final = 1.0
seed = 1
nstep = 9

fs.cosmology_init(omega_m)
ps = fs.PowerSpectrum('../../data/planck_matterpower.dat')

particles = fs.lpt(nc, boxsize, a_init, ps, seed)

fs.pm_init(nc_pm, nc_pm/nc, boxsize)

for i in range(nstep):
    a_vel = a_init + (a_final - a_init)/nstep*(i + 0.5)
    fs.pm_compute_force(particles)
    fs.cola_kick(particles, a_vel)

    a_pos = a_init + (a_final - a_init)/nstep*(i + 1.0)
    fs.cola_drift(particles, a_pos)

filename = 'cola.h5'

particles.save_hdf5(filename, 'ixv')
print('%s written' % filename)

fs.comm_mpi_finalise()
