DIRS = lib py

.PHONY: lib py test clean check
all: $(DIRS)

#
# Set compilers and options
#
CC      := mpic++ -std=c++11
CXX     := mpic++ -std=c++11
OPT     := -DDOUBLEPRECISION

CFLAGS  := $(OPT)


export CC CXX CFLAGS OPT



lib:
	cd lib && $(MAKE) lib

py:
	cd py && $(MAKE) py

test:
	cd test && $(MAKE)

libtest:
	cd lib/test && $(MAKE) libtest

clean:
	for dir in $(DIRS); do (cd $$dir && $(MAKE) clean); done

check:
	cd py && $(MAKE) check
	cd test && $(MAKE) check
