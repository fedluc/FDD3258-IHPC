#include <stdio.h>
#define NB 1
#define TPB 256

// Kernel to print thread id
__global__ void printid(){

  int myid = blockIdx.x * blockDim.x + threadIdx.x; 
  printf("Hello world! My threadId is %d\n", myid);

}

int main(){

  // Launch kernel to print thread id
  printid<<<NB, TPB>>>();
  cudaDeviceSynchronize();

  return 0;

}
