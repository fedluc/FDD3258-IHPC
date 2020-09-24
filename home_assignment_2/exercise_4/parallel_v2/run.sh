#!/bin/bash

#SBATCH -J HA2_ex4
#SBATCH -t 00:10:00
#SBATCH -A edu20.fdd3258
#SBATCH -n 1
#SBATCH -C Haswell

srun -n 1 ./dftw > out.dftw
