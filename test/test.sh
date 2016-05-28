#!/bin/sh

#
# Run python script with mpirun -n 1..4
#

if [ $# -lt 1 ]; then
  echo "sh test.sh <test python file>"
  return 1
fi

for np in 1 2 3 4
do
  mpirun -n $1 || exit 1
done



