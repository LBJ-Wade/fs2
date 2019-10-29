#ifndef FOF_H
#define FOF_H 1

#include <vector>

void fof_find_groups(Particles* const particles, const Float linking_length,
		     Float const * const boxsize3, const int quota=32);
size_t fof_ngroups();
std::vector<Index>& fof_nfof();
std::vector<Index>& fof_grp();
std::vector<Index>& fof_compute_nfof();

#endif
