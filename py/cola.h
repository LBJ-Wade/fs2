#ifndef COLA_H
#define COLA_H 1

#include <vector>
#include "particle.h"

void cola_kick(Particles* const particles, const double a_vel1);
void cola_drift(Particles* const particles, const double a_pos1);
std::vector<Float> cola_velocity(Particles const * const particles);

#endif
