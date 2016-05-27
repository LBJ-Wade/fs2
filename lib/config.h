//
// Definitions of basic variables
//

#ifndef CONFIG_H
#define CONFIG_H 1

#include <math.h>
#include <fftw3-mpi.h>
#include <hdf5.h>

#ifdef DOUBLEPRECISION
typedef double Float;
typedef fftw_complex complex_t;
#define FLOAT_EPS 1.0e-15
#define FLOAT_TYPE MPI_DOUBLE
#define FLOAT_SAVE_TYPE H5T_IEEE_F64LE
#define FLOAT_MEM_TYPE H5T_NATIVE_DOUBLE
#define PRECISION "double"

#else
typedef fftwf_complex complex_t;
typedef float Float;
#define FLOAT_EPS       1.0e-7f
#define FLOAT_TYPE      MPI_FLOAT
#define FLOAT_SAVE_TYPE H5T_IEEE_F32LE
#define FLOAT_MEM_TYPE  H5T_NATIVE_FLOAT
#define PRECISION "single"

#endif

typedef Float Float3[3];

#ifndef M_PI
#define M_PI      3.14159265358979323846264338327950288
#endif


// Memory alignment for SIDM instructions, see
// Section 3.1 SIMD alignment and fftw_malloc in FFTW3 manulal, and
// FFTW3 kernel/align.c source code
#define ALGN 16

size_t size_align(size_t size);

#endif
