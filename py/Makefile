all: build install
#
# Set compilers and options
#
CC      := mpic++ -std=c++11
CXX     := mpic++ -std=c++11

# Extra compile options
OPT     :=
#OPT     += -DUSEHDF5
#OPT     += -DDOUBLEPRECISION

#
# Define library locations if they are not in statndard path
#
FFTW3_DIR ?= #/opt/fftw3 #e.g. /Users/jkoda/Research/opt/gcc/fftw3
GSL_DIR   ?= #e.g. /Users/jkoda/Research/opt/gcc/gsl
HDF5P_DIR ?= # parallel HDF5 library; e.g., brew install hdf5 --with-mpi

DIR_PATH   = $(FFTW3_DIR) $(GSL_DIR) $(HDF5P_DIR)

IDIRS    += $(foreach dir, $(DIR_PATH), $(dir)/include)
LDIRS    += $(foreach dir, $(DIR_PATH), $(dir)/lib)

LIBS    := m gsl gslcblas hdf5

ifeq (,$(findstring -DDOUBLEPRECISION, $(OPT)))
  # Single precision FFTW
  FFTWSUF=f
endif
LIBS += fftw3$(FFTWSUF) fftw3$(FFTWSUF)_mpi

ifdef OPENMP
  LIBS += fftw3$(FFTWSUF)_omp
  #LIBS += fftw3$(FFTWSUF)_threads # for thread parallelization instead of omp
endif

export CC CXX OPT IDIRS LDIRS LIBS

.PHONY: clean build install

build: $(SRC)
	python3 setup.py config build_ext --inplace

install:
	python3 setup.py install

clean:
	rm -rf build dist fs.egg-info

