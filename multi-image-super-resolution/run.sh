#!/bin/bash

#SBATCH --job-name=multi   # job name
#SBATCH --output=multi.out # output log file

#SBATCH --error=gpu.err  # error file
#SBATCH --time=01:00:00  # wall time
#SBATCH --partition=broadwl-lc
#SBATCH --ntasks=1       # 1 CPU core to drive GPU


# Add lines here to run your GPU-based computations.

python -u master.py  --exp_name multi-image1
