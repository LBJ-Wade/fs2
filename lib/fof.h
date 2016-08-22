#ifndef FOF_H
#define FOF_H 1

#include <vector>

void fof_find_group(Particles* const particles, const Float linking_length, const int quota=32);
size_t fof_ngroups();
std::vector<Index>& fof_nfof();

#endif
