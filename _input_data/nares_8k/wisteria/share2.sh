#!/bin/sh
#PJM -L rscgrp=share
#PJM -L elapse=20:00:00
#PJM -L gpu=2
#PJM -g gm42
#PJM -j
#PJM -o share.txt

module load gcc/12.2.0
module load cuda/12.6

cd /work/gm42/m42000/projects/build-v2
./cplate nares_strait_8k/simulation_wisteria.json
