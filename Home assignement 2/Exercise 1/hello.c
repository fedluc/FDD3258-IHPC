#include <stdio.h>
#include <omp.h>

int main(void) {

  omp_set_num_threads(4);
  #pragma omp parallel
  {
    int thread_id = omp_get_thread_num();
    printf("Hello world from Thread %d!\n", thread_id);
  }

  return 0;
  
}
