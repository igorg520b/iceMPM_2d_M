#!/bin/sh
#PJM -L rscgrp=regular-a
#PJM -L node=1
#PJM -L elapse=08:00:00
#PJM -g gm42
#PJM -j
#PJM -o regular_cb8k.txt

module load gcc/12.2.0
module load cuda/12.6

cd /work/gm42/m42000/projects/build-M_v2
./cm2m input/cb8k/cb8k_w.json -r output/cb8k/snapshots/s00000.h5
