#ifndef FFT_H
#define FFT_H 1

#include <stdbool.h>
#include "config.h"
#include "mem.h"

class FFT {
 public:
  FFT(const char name[], const int nc, Mem* mem, const int transposed);
  ~FFT();
  void execute_forward();
  void execute_inverse();
  
  char*       name;
  int         nc;
  float_t*    fx;
  complex_t*  fk;
  ptrdiff_t   local_nx, local_ix0;
  ptrdiff_t   local_nky, local_iky0;
  fftwf_plan  forward_plan, inverse_plan;
  ptrdiff_t   ncomplex;
  bool        allocated;
};

size_t fft_mem_size(const int nc, const int transposed);
size_t fft_local_nx(const int nc);
  
void fft_finalize();
void* fft_malloc(size_t size);

#endif
