#ifndef PM_H
#define PM_H 1

#include "mem.h"
#include "particle.h"

void pm_init(const int nc_pm, const double pm_factor, Mem* const mem_density, Mem* const mem_force, const float_t boxsize);
void pm_compute_force(Particles* const particles);

#endif