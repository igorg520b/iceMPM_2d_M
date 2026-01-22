#!/bin/sh
#PJM -L rscgrp=share-debug
#PJM -L elapse=00:30:00
#PJM -L gpu=2
#PJM -g gm42
#PJM -j
#PJM -o debug2.txt

module load gcc/12.2.0
module load cuda/12.6

cd /work/gm42/m42000/projects/build-v2
./cplate nares_strait_8k/simulation_wisteria.json
