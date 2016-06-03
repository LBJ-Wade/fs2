DIRS = lib py

.PHONY: lib py clean
all: $(DIRS)

#
# Set compilers and options
#
CC      := mpicc
CXX     := mpic++
OPT     := -DDOUBLEPRECISION

CFLAGS  := $(OPT)


export CC CXX CFLAGS OPT



lib:
	cd lib && $(MAKE) lib

py:
	cd py && $(MAKE) py

clean:
	for dir in $(DIRS); do (cd $$dir && $(MAKE) clean); done

