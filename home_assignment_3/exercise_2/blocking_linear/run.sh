#!/bin/bash

#SBATCH -J HA2_ex1
#SBATCH -t 00:01:00
#SBATCH -A edu20.fdd3258
#SBATCH -n 128
#SBATCH -C Haswell

srun -n 8 ./pi
srun -n 16 ./pi 
srun -n 32 ./pi 
srun -n 64 ./pi 
srun -n 128 ./pi
