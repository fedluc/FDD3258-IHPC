#!/bin/bash

#SBATCH -J HA2_ex3
#SBATCH -t 00:10:00
#SBATCH -A edu20.fdd3258
#SBATCH -n 1
#SBATCH -C Haswell

srun -u -n 1 ./maxloc > out.maxloc
