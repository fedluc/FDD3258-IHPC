#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#define N_vec 1000000
#define N_rep 20

void init_array(double *xx);
void find_max_serial( double *xx, int verbose);
void find_max_parallel_naive( double *xx, int N_threads, int verbose);
void find_max_parallel_critical( double *xx, int N_threads, int verbose);
void find_max_parallel_arrays( double *xx, int N_threads, int verbose);
void find_max_parallel_padding( double *xx, int N_threads, int verbose);
void exec_stat( double *xx, int N_threads);

int main(void){

  double start_time;
  int ii,jj;
  double xx[N_vec], tex[N_rep];
  int N_threads[] = {1,2,4,8,16,20,24,28,32};
  size_t len_N_treads = sizeof(N_threads)/ sizeof(N_threads[0]);;

  // init array
  init_array(xx);

  // Check that different versions of the code find the same maximum
  printf("--- Result of the serial code: ");
  find_max_serial(xx,1);
  printf("--- Result  of the parallel code (not protected): ");
  find_max_parallel_naive(xx,32,1);
  printf("--- Result of the parallel code (protected with critical): ");
  find_max_parallel_critical(xx,32,1);
  printf("--- Result of the parallel code (protected with arrays): ");
  find_max_parallel_arrays(xx,32,1);
  printf("--- Result of the parallel code (protected with arrays, optimized with padding): ");
  find_max_parallel_padding(xx,32,1);
  
  // Measure performance
  printf("\n--- Performance of the serial code\n(threads, execution time (ave.) [s], execution time (std. dev.) [s]\n");
  for (ii = 0; ii < N_rep; ii++){
      start_time = omp_get_wtime();
      find_max_serial(xx,0);
      tex[ii] = omp_get_wtime() - start_time;
  }
  exec_stat(tex,1);

  printf("\n--- Performance of the parallel code (not protected)\n(threads, execution time (ave.) [s], execution time (std. dev.) [s]\n");
  for (ii = 0; ii < N_rep; ii++){
      start_time = omp_get_wtime();
      find_max_parallel_naive(xx,32,0);
      tex[ii] = omp_get_wtime() - start_time;

  }
  exec_stat(tex,32);

  printf("\n--- Performance of the parallel code (protected with critical)\n(threads, execution time (ave.) [s], execution time (std. dev.) [s]\n");
  for (jj = 0; jj < len_N_treads; jj++) {
    for (ii = 0; ii < N_rep; ii++){
      start_time = omp_get_wtime();
      find_max_parallel_critical(xx,N_threads[jj],0);
      tex[ii] = omp_get_wtime() - start_time;
      }
    exec_stat(tex, N_threads[jj]);
    
  }

  printf("\n--- Performance of the parallel code (protected with arrays)\n(threads, execution time (ave.) [s], execution time (std. dev.) [s]\n");
  for (jj = 0; jj < len_N_treads; jj++) {
    for (ii = 0; ii < N_rep; ii++){
      start_time = omp_get_wtime();
      find_max_parallel_arrays(xx,N_threads[jj],0);
      tex[ii] = omp_get_wtime() - start_time;
      }
    exec_stat(tex, N_threads[jj]);
    
  }

  printf("\n --- Performance of the parallel code (protected with arrays, optimized with padding)\n(threads, execution time (ave.) [s], execution time (std. dev.) [s]\n");
  for (jj = 0; jj < len_N_treads; jj++) {
    for (ii = 0; ii < N_rep; ii++){
      start_time = omp_get_wtime();
      find_max_parallel_padding(xx,N_threads[jj],0);
      tex[ii] = omp_get_wtime() - start_time;
      }
    exec_stat(tex, N_threads[jj]);
    
  }
   
  
  return 0;

}

// function to initialize the array
void init_array( double *xx){

  srand(time(0));
  for(int ii=0; ii < N_vec; ii++){
   xx[ii] = ((double)(rand()) / RAND_MAX)*
            ((double)(rand())/ RAND_MAX)*
            ((double)(rand()) / RAND_MAX)*1000;
  }
  
}

// function to locate the maximum in the array (serial version)
void find_max_serial( double *xx, int verbose) {

  double maxval = xx[0];
  int maxloc = 0;
  
  for (int ii=0; ii < N_vec; ii++){
    if (xx[ii] > maxval) {
      maxval = xx[ii];
      maxloc = ii;
    }
  }

  if(maxval < 0){
    printf("Error, maxval out of bound\n");
  }
  
  if (verbose == 1){
      printf("Maximum %.8f at position %d\n", maxval, maxloc);
  }
  
}

// function to locate the maximum in the array (parallel version, not protected)
void find_max_parallel_naive( double *xx, int N_threads, int verbose) {

  double maxval = -1.0e-30;
  int maxloc = 0;

  omp_set_num_threads(N_threads);
  #pragma omp parallel for
  for (int ii=0; ii < N_vec; ii++){
    if (xx[ii] > maxval) {
      maxval = xx[ii];
      maxloc = ii;
    }
  }



  if (verbose == 1){
      printf("Maximum %.8f at position %d\n", maxval, maxloc);
  }
  
}

// function to locate the maximum in the array (parallel version, protected with critical)
void find_max_parallel_critical( double *xx, int N_threads, int verbose) {

  double maxval = -1.0e-30;
  int maxloc = 0;

  omp_set_num_threads(N_threads);
  #pragma omp parallel for
  for (int ii=0; ii < N_vec; ii++){
    #pragma omp critical
    {
      if (xx[ii] > maxval) {
        maxval = xx[ii];
        maxloc = ii;
      }
    }
  }

  if (verbose == 1){
      printf("Maximum %.8f at position %d\n", maxval, maxloc);
  }
  
}

// function to locate the maximum in the array (parallel version, protected with arrays)
void find_max_parallel_arrays( double *xx, int N_threads, int verbose) {

  double *maxval_thread, maxval;
  int *maxloc_thread, maxloc;

  maxval_thread = (double*)malloc(sizeof(double) * N_threads);
  maxloc_thread = (int*)malloc(sizeof(int) * N_threads);
  
  omp_set_num_threads(N_threads);
  #pragma omp parallel shared(maxval_thread,maxloc_thread)
  {
    int id = omp_get_thread_num();
    maxval_thread[id] = -1.0e-30;
    #pragma omp for
      for (int ii=0; ii < N_vec; ii++){
	if (xx[ii] > maxval_thread[id]) {
	  maxval_thread[id] = xx[ii];
	  maxloc_thread[id] = ii;
	}
      }
  }

  maxval = maxval_thread[0];
  maxloc = maxloc_thread[0];

  for (int ii=0; ii < N_threads; ii++){
    if (maxval_thread[ii] > maxval) {
      maxval = maxval_thread[ii];
      maxloc = maxloc_thread[ii];
    }
  }
  free(maxval_thread);
  free(maxloc_thread);
  
  if (verbose == 1){
      printf("Maximum %.8f at position %d\n", maxval, maxloc);
  }


}


// function to locate the maximum in the array (parallel version, protected with arrays)
typedef struct {double val; int loc; char pad[128];} tvals;

void find_max_parallel_padding( double *xx, int N_threads, int verbose) {

  double maxval;
  int maxloc;
  tvals *maxinfo_thread;
  
  maxinfo_thread = (tvals*)malloc(sizeof(tvals) * N_threads);
  
  omp_set_num_threads(N_threads);
  #pragma omp parallel shared(maxinfo_thread)
  {
    int id = omp_get_thread_num();
    maxinfo_thread[id].val = -1.0e-30;
    #pragma omp for
      for (int ii=0; ii < N_vec; ii++){
	if (xx[ii] > maxinfo_thread[id].val) {
	  maxinfo_thread[id].val = xx[ii];
	  maxinfo_thread[id].loc = ii;
	}
      }
  }

  maxval = maxinfo_thread[0].val;
  maxloc = maxinfo_thread[0].loc;

  for (int ii=0; ii < N_threads; ii++){
    if (maxinfo_thread[ii].val > maxval) {
      maxval = maxinfo_thread[ii].val;
      maxloc = maxinfo_thread[ii].loc;
    }
  }
  free(maxinfo_thread);

  if (verbose == 1){
      printf("Maximum %.8f at position %d\n", maxval, maxloc);
  }


}


// function to compute the statistics of the execution time
void exec_stat( double *xx, int N_threads) {

  double ave=0, std_dev=0;

  for (int ii=0; ii < N_rep; ii++){
    ave += xx[ii]/N_rep;
  }

  for (int ii=0; ii < N_rep; ii++){
    std_dev += (xx[ii]-ave)*(xx[ii]-ave)/N_rep;
  }
  std_dev = sqrt(std_dev);
  printf("%d %.5e %.5e \n", N_threads, ave, std_dev);


}
