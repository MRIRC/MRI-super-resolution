#!/bin/bash

#SBATCH --job-name=superres   # job name
#SBATCH --output=superres.out # output log file

#SBATCH --error=gpu.err  # error file
#SBATCH --time=02:00:00  # wall time
#SBATCH --nodes=1        # 1 GPU node
#SBATCH --partition=gpu2 # GPU2 partition
#SBATCH --ntasks=1       # 1 CPU core to drive GPU
#SBATCH --gres=gpu:1     # Request 1 GPU

# Load all required modules below. As an example we load cuda/9.1
#module unload cuda
module load cuda/11.0

# Add lines here to run your GPU-based computations.

python -u master.py --total_steps $1 --seg $2 --hidden_layers $3 --hidden_features $4 --learning_rate $5 --exp_name $6 --erd $7 > $6.out
