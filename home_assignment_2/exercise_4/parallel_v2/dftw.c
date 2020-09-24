    // the code calculates a DFT of a random complex number input and 
    // then an IDFT. The IDFT result should be the input vector 
    // to compile with gcc
    // gcc -Wall -O2 -fopenmp -o DFTW DFTW.c 
    // written by stef

    // exercise


	#include "stdio.h" // printf
	#include "stdlib.h" // malloc and rand for instance. Rand not thread safe!
	#include "time.h"   // time(0) to get random seed
	#include "math.h"  // sine and cosine
	#include "omp.h"   // openmp library like timing

	
	// two pi
	#define PI2 6.28318530718
	// this for the rounding error, increasing N rounding error increases
        // 0.01 precision good for N > 8000
	#define R_ERROR 0.01

	// main routine to calculate DFT
        int DFT(int idft, double* xr, double* xi, double* Xr_o, double* Xi_o, int N, int N_threads, int verbose);
	// set the input array with random number
	int fillInput(double* xr, double* xi, int N);
	// set to zero the input vector
	int setOutputZero(double* Xr_o, double* Xi_o, int N);
	// check if x = IDFT(DFT(x))
	int checkResults(double* xr, double* xi, double* xr_check, double* xi_check, double* Xr_o, double* Xi_r, int N);
	// print the results of the DFT
	int printResults(double* xr, double* xi, int N);
        // compute statistics of execution time
        void exec_stat( double *xx, int N_threads, int N_rep);


	int main(int argc, char* argv[]){
	  
	  // size of input array
	  int N = 10000; // 8,000 is a good number for testing
	  printf("DFTW calculation with N = %d \n",N);

	  // number of threads to test
	  int N_threads[] = {1,2,4,8,16,20,24,28,32};
	  size_t len_N_threads = sizeof(N_threads)/ sizeof(N_threads[0]);

	  // Number of repetitions of the test used to collect statistics
	  int N_rep = 20;
	  double* tex = (double*) malloc (N_rep *sizeof(double));
	  
	  // Allocate array for input vector
	  double* xr = (double*) malloc (N *sizeof(double));
	  double* xi = (double*) malloc (N *sizeof(double));

	  // for checking purposes
	  double* xr_check = (double*) malloc (N *sizeof(double));
	  double* xi_check = (double*) malloc (N *sizeof(double));
	  
	  // Allocate array for output vector
	  double* Xr_o = (double*) malloc (N *sizeof(double));
	  double* Xi_o = (double*) malloc (N *sizeof(double));

	  // Run once to check if the result is good
	  int idft;
	  double start_time;
          printf("Check result of DFT\n");	  
	  for (int i=0; i<len_N_threads; i++) {

	      // Initialize input and output
	      fillInput(xr,xi,N);
	      setOutputZero(xr_check,xi_check,N);
	      setOutputZero(Xr_o,Xi_o,N);
	 
	      // DFT
	      idft = 1;	      
	      DFT(idft,xr,xi,Xr_o,Xi_o,N, N_threads[i],1);

	      // IDFT
	      idft = -1;
	      DFT(idft,Xr_o,Xi_o,xr_check,xi_check,N, N_threads[i],0);	  
	    
	      // check the results: easy to make correctness errors with openMP
	      checkResults(xr,xi,xr_check,xi_check,Xr_o, Xi_o, N);

	      // print the results of the DFT
              #ifdef DEBUG
	      printResults(Xr_o,Xi_o,N);
              #endif
	      
	  }
	  
	  // Run several times to measure the execution time
          printf("Execution time: average and standard deviation in seconds\n");	  
	  for (int i=0; i<len_N_threads; i++) {

	    for (int j=0; j<N_rep; j++){
	      
	      // Initialize input and output
	      fillInput(xr,xi,N);
	      setOutputZero(xr_check,xi_check,N);
	      setOutputZero(Xr_o,Xi_o,N);
	  
	      // start timer
	      start_time = omp_get_wtime();	  

	      // DFT
	      idft = 1;	      
	      DFT(idft,xr,xi,Xr_o,Xi_o,N, N_threads[i], 0);

	      // IDFT
	      idft = -1;
	      DFT(idft,Xr_o,Xi_o,xr_check,xi_check,N, N_threads[i], 0);	  
	    
	      // stop timer
	      tex[j] =  omp_get_wtime() - start_time;
	      
	    }
	    exec_stat(tex, N_threads[i], N_rep);

	  }
	    
	  // take out the garbage
	  free(tex);
	  free(xr); free(xi);
	  free(Xi_o); free(Xr_o);
	  free(xr_check); free(xi_check);
	  
	  return 1;
	}

        // DFT/IDFT routine
        // idft: 1 direct DFT, -1 inverse IDFT (Inverse DFT)
        int DFT(int idft, double* xr, double* xi, double* Xr_o, double* Xi_o, int N, int N_threads, int verbose){

          omp_set_num_threads(N_threads);

          #pragma omp parallel
          {


            #pragma omp single
            {
              if (verbose == 1)
                printf("Number of threads: %d\n", omp_get_num_threads());
            }

            #pragma omp for
              for (int k=0 ; k<N ; k++) {
                for (int n=0 ; n<N ; n++)  {
                    // Real part of X[k]
                    Xr_o[k] += xr[n] * cos(n * k * PI2 / N) + idft*xi[n]*sin(n * k * PI2 / N);
                    // Imaginary part of X[k]
                    Xi_o[k] += -idft*xr[n] * sin(n * k * PI2 / N) + xi[n] * cos(n * k * PI2 / N);
                }
              }

          }

          // normalize if you are doing IDFT
          if (idft==-1){
            for (int n=0 ; n<N ; n++){
              Xr_o[n] /=N;
              Xi_o[n] /=N;
            }
          }

          return 1;
        }

	
	// set the initial signal 
    // be careful with this 
    // rand() is NOT thread safe in case
	int fillInput(double* xr, double* xi, int N){
	    srand(time(0));
	    for(int n=0; n < 100000;n++) // get some random number first 
	    	rand();
	    for(int n=0; n < N;n++){
	       // Generate random discrete-time signal x in range (-1,+1)
	       //xr[n] = ((double)(2.0 * rand()) / RAND_MAX) - 1.0;
	       //xi[n] = ((double)(2.0 * rand()) / RAND_MAX) - 1.0;
	       // constant real signal
	       xr[n] = 1.0;
	       xi[n] = 0.0;
	    }
		return 1; 
	}

	// set to zero the output vector
	int setOutputZero(double* Xr_o, double* Xi_o, int N){
	for(int n=0; n < N;n++){
	       Xr_o[n] = 0.0;
	       Xi_o[n] = 0.0; 
	    }
		return 1;
	}

	// check if x = IDFT(DFT(x)) 
	int checkResults(double* xr, double* xi, double* xr_check, double* xi_check, double* Xr_o, double* Xi_r, int N){
		// x[0] and x[1] have typical rounding error problem
		// interesting there might be a theorem on this
		for(int n=0; n < N;n++){
			if (fabs(xr[n] - xr_check[n]) > R_ERROR)
			    printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n",n,xr[n], n,xr_check[n]);
			if (fabs(xi[n] - xi_check[n]) > R_ERROR)
			    printf("ERROR - x[%d] = %f, inv(X)[%d]=%f \n",n,xi[n], n,xi_check[n]);

		}
		printf("Xre[0] = %f \n",Xr_o[0]);
		return 1;
	}

	// print the results of the DFT
	int printResults(double* xr, double* xi, int N){
		for(int n=0; n < N;n++)
			    printf("Xre[%d] = %f, Xim[%d] = %f \n", n, xr[n], n, xi[n]);
		return 1;
	}

        // function to compute the statistics of the execution time
        void exec_stat( double *xx, int N_threads, int N_rep) {

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

