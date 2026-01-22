#!/bin/sh
#PJM -L rscgrp=short-o
#PJM -L elapse=02:00:00
#PJM -L node=1
#PJM -g gm42
#PJM -j
#PJM --mpi proc=48
#PJM -o compress_parallel.txt

module load odyssey
module load hdf5/1.12.0

cd /work/gm42/m42000/projects/build-c2
mpiexec compressor_mpi /work/gm42/m42000/projects/test/output 1 53 --overwrite
