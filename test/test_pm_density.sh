#!/bin/sh

for n in 1 2 3 4
do
  echo "#"
  echo "# n_node = $n"
  echo "#"
  mpirun -n $n python3 test_pm_density.py || exit 1
done
  
