#ifndef PM_H
#define PM_H 1

#include "fft.h"
#include "mem.h"
#include "particle.h"

enum class PmStatus {density_done, force_done, done};


void pm_init(const int nc_pm, const double pm_factor, Mem* const mem_density, Mem* const mem_force, const Float boxsize);
void pm_free();

void pm_compute_force(Particles* const particles);
FFT* pm_compute_density(Particles* const particles);
void pm_check_total_density();

FFT* pm_get_fft();

PmStatus pm_get_status();
void pm_set_status(PmStatus pm_status);

#endif
