#ifndef LEAPFROG_H
#define LEAPFROG_H 1

#include "particle.h"

void leapfrog_set_initial_velocity(Particles* const particles, const double a);
void leapfrog_kick(Particles* const particles, const double avel1);
void leapfrog_drift(Particles* const particles, const double apos1);

#endif
