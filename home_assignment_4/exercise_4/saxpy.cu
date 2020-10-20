#include <stdio.h>
#include <sys/time.h>

// Kernel to print thread id
__global__ void saxpyGPU(float *xx, float *yy, float aa){

  int ii = blockIdx.x * blockDim.x + threadIdx.x;
  yy[ii] += aa * xx[ii];

}

int main(){

  // Array size
  #define ARRAY_SIZE 10000
  printf("Array size: %d\n", ARRAY_SIZE);

  // Threads per block
  #define TPB 256

  // arrays for saxpy
  float xx[ARRAY_SIZE], yy[ARRAY_SIZE];
  float resGPU[ARRAY_SIZE], resCPU[ARRAY_SIZE];
  xx[0] = 0.0;
  yy[0] = 0.0;
  for (int ii = 1; ii < ARRAY_SIZE; ii++) {
      xx[ii] = xx[ii-1] + 1.0;
      yy[ii] = yy[ii-1] + 2.0;
  }

  // Constant for saxpy
  float aa = 2.5;


  // ------ CPU calculations ------

  // Start timing
  printf("Computing SAXPY on the CPU...");
  struct timeval start, end;
  gettimeofday(&start, NULL);

  // Compute saxpy
  for (int ii = 0; ii < ARRAY_SIZE; ii++) {
      resCPU[ii] = xx[ii]*aa + yy[ii];
  }

  // End timing
  gettimeofday(&end, NULL);
  printf("Done. Elapsed time: %ld microseconds\n",
         ((end.tv_sec * 1000000 + end.tv_usec) -
          (start.tv_sec * 1000000 + start.tv_usec)));

  // ------ GPU calculations ------

  // Start timing
  printf("Computing SAXPY on the GPU...");
  gettimeofday(&start, NULL);

  // Allocate arrays
  float *d_xx = NULL;
  float *d_yy = NULL;
  cudaMalloc(&d_xx, ARRAY_SIZE*sizeof(float));
  cudaMalloc(&d_yy, ARRAY_SIZE*sizeof(float));

  // Transfer data to GPU
  cudaMemcpy(d_xx, xx, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_yy, yy, ARRAY_SIZE*sizeof(float), cudaMemcpyHostToDevice);

  // Compute saxpy
  saxpyGPU<<<(ARRAY_SIZE + TPB -1)/TPB, TPB>>>(d_xx,d_yy,aa);

  // Retrieve data from GPU
  cudaMemcpy(resGPU, d_yy, ARRAY_SIZE*sizeof(float), cudaMemcpyDeviceToHost);

  // Free memory
  cudaFree(d_xx);
  cudaFree(d_yy);

  // End timing
  gettimeofday(&end, NULL);
  printf("Done. Elapsed time: %ld microseconds\n",
         ((end.tv_sec * 1000000 + end.tv_usec) -
          (start.tv_sec * 1000000 + start.tv_usec)));


  // ------ Compare results ------
  printf("Comparing the output for each implementation...");
  int err = 0;
  float maxdiff = powf(10,-8);

  for (int ii = 0; ii < ARRAY_SIZE; ii++) {
    if (fabsf(resCPU[ii] - resGPU[ii]) > maxdiff) err = 1;
  }

  if (err == 0) {
    printf("Correct\n!");
  }
  else {
    printf("Wrong\n!");
  }


  return 0;

}
