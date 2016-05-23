##!/bin/sh

for np in 1 4
do
  mpirun -n $np python3 test_fft.py || exit 1
done
  
