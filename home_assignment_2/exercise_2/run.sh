#!/bin/bash

#SBATCH -J HA2_ex2
#SBATCH -t 00:05:00
#SBATCH -A edu20.fdd3258
#SBATCH -n 1
#SBATCH -C Haswell

MY_THREADS=1
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=2
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=4
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=8
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=12
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=16
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=20
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=24
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=28
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

MY_THREADS=32
export OMP_NUM_THREADS=${MY_THREADS}
srun -n 1 ./stream > out.stream${MY_THREADS}_1
srun -n 1 ./stream > out.stream${MY_THREADS}_2
srun -n 1 ./stream > out.stream${MY_THREADS}_3
srun -n 1 ./stream > out.stream${MY_THREADS}_4
srun -n 1 ./stream > out.stream${MY_THREADS}_5

