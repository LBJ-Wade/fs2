#
# fs: fast cosmological simulation system
#   

#
# Compile options
#

#OPT+= -DDOUBLEPRECISION
OPT += -DCHECK  # slow assersions

#
# Compiler configurations
#

# Define OPENMP to enable MPI+OpenMP hybrid parallelization
# OPENMP  = -fopenmp # -openmp for Intel, -fopenmp for gcc, llvm doesn't support

CXX      = mpic++

WOPT    ?= -Wall
CFLAGS  := -O3 $(WOPT) $(OPENMP) $(OPT)
LIBS    := -lm

# Define paths of FFTW3 & GSL libraries if necessary.

FFTW3_DIR ?= #e.g. /Users/jkoda/Research/opt/gcc/fftw3
GSL_DIR   ?= #e.g. /Users/jkoda/Research/opt/gcc/gsl

DIR_PATH = $(FFTW3_DIR) $(GSL_DIR)

CFLAGS  += $(foreach dir, $(DIR_PATH), -I$(dir)/include)
LIBS    += $(foreach dir, $(DIR_PATH), -L$(dir)/lib)

LIB      = libfs.a libfs.dylib
all: $(LIB)

OBJS := comm.o msg.o config.o fft.o mem.o
OBJS += power.o


#
# Linking libraries
#
# LIBS += -llua -ldl 
LIBS += -lgsl -lgslcblas

ifeq (,$(findstring -DDOUBLEPRECISION, $(OPT)))
  # Single precision FFTW
  FFTWSUF=f
endif
LIBS += -lfftw3$(FFTWSUF) -lfftw3$(FFTWSUF)_mpi

ifdef OPENMP
  LIBS += -lfftw3$(FFTWSUF)_omp
  #LIBS += -lfftw3$(FFTWSUF)_threads # for thread parallelization instead of omp
endif

#
# Library libfs.a
#
libfs.a: $(OBJS)
	ar r $@ $(OBJS)

libfs.dylib: $(OBJS)
	$(CXX) -dynamiclib $(LIBS) $(OBJS) -o $@


.PHONY: clean run dependence
clean:
	rm -f $(EXEC) $(OBJS)

dependence:
	gcc -MM -MG *.c

