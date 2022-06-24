#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#else
#warning CUDA is not available.
#endif

#include <test_inputs.h>
#include <emc_utils.h>

// Version 0.3
//
// First attempt at using AVX intrinsics
//

typedef float real;

int main(int argc, char ** argv) {
  int verbosity = 1;
  int vecSize = N;
  int numThreads = 1;
  int parAvg = 0;
  // Parse command line arguments
  for (int i=1; i<argc; ++i) {
    // Parse options which take an additional parameter
    if ((i+1) < argc) {
      // Specify the size of the input array
      if (!strcmp(argv[i], "-N")) {
        vecSize = atoi( argv[++i] );
      }
      // Specify the number of available threads for parallel execution
      else if (!strcmp(argv[i], "-nt")) {
        numThreads = atoi( argv[++i] );
      }
    }
    // Parse single parameter switches
    else {
      // Verbosity can be increased by repeating '+v'
      if (!strcmp(argv[i], "+v")) {
        verbosity++;
      }
      // Verbosity can be decreased by repeating '-v'
      if (!strcmp(argv[i], "-v")) {
        verbosity--;
      }
    }
  }
  
  // Report input vector size
  if (verbosity > 0) printf("vecSize:    %d\n", vecSize);
  // Report number of threads
  if (verbosity > 0) printf("numThreads: %d\n", numThreads);
  omp_set_num_threads(numThreads);
  
  // Array for recording time-stamps
  int T = 0;
  double t[32];
  
  // NOTE: I'm splitting out the following steps of caching X and
  // computing avg & var so that it's easier to see any latency due to
  // the initial load of data values.
  
  // Cache the input vector x' = x + v + b
  real * X = new real[vecSize];
  t[T++] = omp_get_wtime();
#pragma omp parallel for
  for (int i=0; i<vecSize; ++i) {
    X[i] = x_test[i%N] + v_test[i%N] + b_test[i%N];
  }
  t[T++] = omp_get_wtime();
  if (verbosity > 0) printf("%15.10f \t Time to compute x' = x + v + b.\n", t[T-1]-t[T-2]);

  const real eps = 1e-5;
  const real E = xAvg.mean();
  const real rV = 1.0/xAvg.variance() + eps;
  real * Y = new real[vecSize];
  real t2 = omp_get_wtime();
  for (int i=0; i<vecSize; ++i) {
    Y[i] = (x_test[i%N] + v_test[i%N] + b_test[i%N] - E)*gamma_test[i%N]*rV + beta_test[i%N];
  }
  real t3 = omp_get_wtime();
  
  real t4 = omp_get_wtime();
  for (int i=0; i<vecSize; ++i) {
    Y[i] = (X[i] - E)*gamma_test[i%N]*rV + beta_test[i%N];
  }
  real t5 = omp_get_wtime();

  real l2 = 0.0;
  real t6 = omp_get_wtime();
  for (int i=0; i<vecSize; ++i) {
    real diff = Y[i] - Y[i%N];
    l2 += diff*diff;
  }
  real t7 = omp_get_wtime();
  printf("Sum of squared errors in the output vector Y: %g\n", sqrt(l2/vecSize));

  delete [] X;
  delete [] Y;

  printf("%10.4g \t %6.2f%% \t Time to compute mean and variance (and X)\n", t1-t0, 100*(t1-t0)/(t7-t0));
  printf("%10.4g \t %6.2f%% \t Time to compute output vector (without X)\n", t3-t2, 100*(t3-t2)/(t7-t0));
  printf("%10.4g \t %6.2f%% \t Time to compute output vector (with X)\n", t5-t4, 100*(t5-t4)/(t7-t0));
  printf("%10.4g \t %6.2f%% \t Time to compute L2 error norm\n", t7-t6, 100*(t7-t6)/(t7-t0));
  printf("------------------\n");
  printf("%10.4g \t %6.2f%% \t Total execution time\n", t7-t0, 100.0);
  printf("\n");
  return 0;
}
