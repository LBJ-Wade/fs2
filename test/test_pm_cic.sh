#!/bin/sh

for np in 1 2 3 4
do
  echo "#"
  echo "# test_pm_cic.py with $np nodes"
  echo "#"
  mpirun -n $np python3 test_pm_cic.py || exit 1
done
