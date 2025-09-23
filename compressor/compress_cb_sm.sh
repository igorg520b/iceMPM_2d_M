#!/bin/sh
#PJM -L rscgrp=short-o
#PJM -L elapse=07:20:00
#PJM -L node=1
#PJM -g gm42
#PJM -j
#PJM -o compress.txt

module load odyssey
module load hdf5/1.12.0

cd /work/gm42/m42000/projects/build-compressor
./compressor /work/gm42/m42000/projects/build-M_v2/output/cb_alt_w 0 380
