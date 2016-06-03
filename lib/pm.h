#ifndef PM_H
#define PM_H 1

#include "fft.h"
#include "mem.h"
#include "particle.h"

void pm_init(const int nc_pm, const double pm_factor, Mem* const mem_density, Mem* const mem_force, const Float boxsize);
void pm_compute_force(Particles* const particles);
FFT* pm_compute_density(Particles* const particles);

FFT* pm_get_fft();

#endif
