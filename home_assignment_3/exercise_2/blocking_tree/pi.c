#include <stdlib.h>
#include <stdio.h>
#include <mpi.h>
#include <math.h>
#include <time.h>

int pi_mpi(int argc, char *argv[], int flip, double *out);
int pi_local(int flip, int rank);

	      
int main(int argc, char *argv[]) {

  int local_count, tmp_count, flip = 1 << 24;
  int rank, num_ranks, provided, bin_step_rec, bin_step_send, bin_ranks;
  //double pi;
  
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

  // Binary tree reduction
  bin_step_rec = 2;
  bin_step_send = 1;
  bin_ranks = num_ranks;

  while ( bin_ranks > 1 ){
 
      // Receive data
      if ( rank % bin_step_rec == 0 ) {
	MPI_Recv(&tmp_count, 1, MPI_INT, rank + bin_step_send,
	       0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      }
      else if ( rank % bin_step_send == 0 && rank % bin_step_rec != 0 ){
	MPI_Send(&local_count, 1, MPI_INT, rank - bin_step_send, 0, MPI_COMM_WORLD);     
      }

  

  // Update count
  local_count += tmp_count;

  // Update parameters for tree reduction
  bin_step_rec *= 2;
  bin_step_send *= 2;
  bin_ranks /= 2;
  
  }
 
  // End timing
  end_time = MPI_Wtime();

  // Output
  if (rank == 0) {
    printf("--------------------\nNumber of MPI ranks: %d\n",num_ranks);
    printf("pi = %.8e\n", ((double) local_count / (double) flip) * 4.0);
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
