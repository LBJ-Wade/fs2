DIRS = lib py

.PHONY: lib py test clean
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

clean:
	for dir in $(DIRS); do (cd $$dir && $(MAKE) clean); done

