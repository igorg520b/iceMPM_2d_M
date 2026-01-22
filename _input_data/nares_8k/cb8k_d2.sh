#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L elapse=00:05:00
#PJM -L gpu=2
#PJM -g gm42
#PJM -j
#PJM -o debug_cb8k.txt

module load gcc/12.2.0
module load cuda/12.6

cd /work/gm42/m42000/projects/build-M_v2
./cm2m input/cb8k/cb8k_dw2.json -r output/cb8k/snapshots/s00000.h5
