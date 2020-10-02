#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {

  int rank, size, provided;

  // Initialize MPI
  MPI_Init_thread(&argc, &argv, MPI_THREAD_SINGLE, &provided);

  // Get total number of processes
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // Get local rank
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  // Output on screen
  printf("Hello world from rank %d from %d processes!\n", rank, size);

  // Finalize MPI
  MPI_Finalize();

  return 0;
  
}
