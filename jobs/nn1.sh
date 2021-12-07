#!/bin/bash
#SBATCH -p miriel
#SBATCH -N 1
#SBATCH --job-name=nn1
#SBATCH --mem=10G --time=1-0:00
#SBATCH -o slurm_out/nn1.out
#SBATCH -e slurm_out/nn1.err

cd /home/chp-truel/projet_NN

module purge
module load language/python/3.8.0
module load compiler/gcc/9.3.0
module load mpi/openmpi/4.0.1
module load compiler/cuda/10.0

python3.8 src/NN1.py 4 6 8
