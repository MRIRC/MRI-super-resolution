#!/bin/bash

#SBATCH --job-name=super_erd   # job name
#SBATCH --output=super_erd.out # output log file

#SBATCH --error=gpu.err  # error file
#SBATCH --time=02:00:00  # wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

# Load all required modules below. As an example we load cuda/9.1
#module unload cuda
module load cuda/10.2

# Add lines here to run your GPU-based computations.

python -u master.py --erd  --exp_name sr1
