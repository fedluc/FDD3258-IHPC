#!/bin/bash

#SBATCH -J HA2_ex1
#SBATCH -t 00:01:00
#SBATCH -A edu20.fdd3258
#SBATCH -n 4
#SBATCH -C Haswell

srun -n 4 ./hello > out.hello
