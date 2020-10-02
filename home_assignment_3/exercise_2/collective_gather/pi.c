#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

int pi_mpi(int argc, char *argv[], int flip, double *out);
int pi_local(int flip, int rank);

	      
int main(int argc, char *argv[]) {

  int local_count, ii, flip = 1 << 24;
  int rank, num_ranks, provided;
  double pi;
  
  // Initialize MPI
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  // Start timing
  double start_time, end_time;
  start_time = MPI_Wtime();
  
  // Info about processes and local rank
  MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Update local count on each process
  local_count = pi_local(flip/num_ranks, rank);

  // Gather data to root
  int *counts = NULL;
  if ( rank == 0 ) {
    counts = malloc(sizeof(int) * (num_ranks - 1));
  }
  MPI_Gather(&local_count, 1, MPI_FLOAT, counts, 1, MPI_FLOAT, 0,
           MPI_COMM_WORLD);

  // Compute pi
  if ( rank == 0) {

    // Update global count
    int global_count = local_count;
    for (ii = 0; ii < num_ranks - 1; ii++){
      global_count += counts[ii];
    }

    // Compute pi
    pi = ((double) global_count / (double) flip) * 4.0;

    // Clean memory
    free(counts);
      
  }
  
  // End timing
  end_time = MPI_Wtime();

  // Output
  if (rank == 0) {
    printf("--------------------\nNumber of MPI ranks: %d\n",num_ranks);
    printf("pi = %.8e\n", pi);
    printf("Elapsed time : %.8e\n", end_time - start_time);
  }
  
  // Finalize MPI
  MPI_Finalize();

  return 0;
 
  
}


int pi_local(int flip, int rank) {

  int ii, count=0;
  double xx, yy, zz;
 
  // Initialize seed of random number generator
  srand(time(NULL) + 123456789 + rank*100);

  // Monte-Carlo calculation of pi
  for (ii = 0; ii < flip; ii++){

    // Random (x,y) cartesian coordinates
    xx = (double)random() / (double)RAND_MAX;
    yy = (double)random() / (double)RAND_MAX;

    // Radial distance
    zz = sqrt(xx*xx + yy*yy);

    // Update count if the radial distance falls inside the circle of unit radius
    if (zz <= 1.0){
      count++;
    }
  
  }

  return count;
    
}
