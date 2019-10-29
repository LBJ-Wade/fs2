#
# Runs COmoving Lagrangian Acceleration simulation
#
# mpirun -n 1 python3 cola.py
#
# Input:
#     Initial power spectrum: ../data/planck_matterpower.dat
# Output:
#     cola.h5
#
import signal
import fs


signal.signal(signal.SIGINT, signal.SIG_DFL) # enable cancel with ctrl-c


# Parameters
omega_m = 0.308
nc = 64
nc_pm = nc
boxsize = 64
a_init = 0.1
a_final = 1.0
seed = 1
nstep = 9

# Initialisation
fs.cosmology.init(omega_m)
ps = fs.PowerSpectrum('../data/planck_matterpower.dat')

# Initial condition
particles = fs.lpt.init(nc, boxsize, a_init, ps, seed, 'cola')

fs.pm.init(nc_pm, nc_pm/nc, boxsize)

for i in range(nstep):
    a_vel = a_init + (a_final - a_init)/nstep*(i + 0.5)
    fs.pm.force(particles)

    fs.cola.kick(particles, a_vel)

    a_pos = a_init + (a_final - a_init)/nstep*(i + 1.0)
    fs.cola.drift(particles, a_pos)

filename = 'cola.h5'

particles.save_hdf5(filename, 'ixv')
print('%s written' % filename)


