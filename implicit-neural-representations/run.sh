#!/bin/bash

#SBATCH --job-name=quant.test   # job name
#SBATCH --output=quant.out.test # output log file

#SBATCH --error=gpu.err  # error file
#SBATCH --time=01:00:00  # 1 hour of wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

# Load all required modules below. As an example we load cuda/9.1
module unload cuda
module load cuda/10.2

# Add lines here to run your GPU-based computations.

python -u master.py --out_img_folder OUT_IMG_FOLDER  
