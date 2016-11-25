default:
	cd ../.. && $(MAKE) libtest

TESTS := test_kdtree

libtest: $(TESTS)



VPATH := ..
CXXFLAGS := -I.. -O3 $(OPENMP) $(OPT)
LIBS  := -L.. -lfs

#../libfs.dylib:
#	cd .. && $(MAKE)

OBJS := test_kdtree.o

test_kdtree: test_kdtree.o
	$(CXX) $^ -o $@

test_kdtree.o: test_kdtree.cpp fs.h

%.cpp: ../%.cpp
	ln -s $< .

$(TESTS): ../libfs.dylib


.PHONY: clean test
clean:
	rm -f $(LIB) $(OBJS)

test:
	echo "test"
	./test_kdtree
