#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#include <test_inputs.h>
#include <emc_utils.h>

// Version 0.1.0
//
// This version will serve as the baseline for comparision against
// future optimized versions.
//
// All loops are serial.
//
// Only a few modest attempts at efficiency have been implemented.
//
int main(int argc, char ** argv) {
  bool verbose = false;
  int vecSize = N;
  int numThreads = 1;
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
      if (!strcmp(argv[i], "-v")) {
        verbose = true;
      }
    }
  }
  // Report input vector size
  if (verbose) printf("vecSize: %d\n", vecSize);
  
  // This is a running average calculator that I use quite a lot when
  // reading in data from a file or a stream without having to store
  // every value.
  OnlineAverage<double> xAvg;
  double t0 = omp_get_wtime();
  double * X = new double[vecSize];
  for (int i=0; i<vecSize; ++i) {
    X[i] = x_test[i%N] + v_test[i%N] + b_test[i%N];
    xAvg += X[i];
  }
  double t1 = omp_get_wtime();

  if (verbose) {
    printf("xAvg.count: %ld\n", xAvg.count());
    printf("xAvg.mean: %f\n", xAvg.mean());
    printf("xAvg.min: %f\n", xAvg.min());
    printf("xAvg.max: %f\n", xAvg.max());
    printf("xAvg.range: %f\n", xAvg.range());
    printf("xAvg.variance: %f\n", xAvg.variance());
    printf("xAvg.stdDev: %f\n", xAvg.stdDev());
  }

  const double eps = 1e-5;
  const double E = xAvg.mean();
  const double rV = 1.0/xAvg.variance() + eps;
  double * Y = new double[vecSize];
  double t2 = omp_get_wtime();
  for (int i=0; i<vecSize; ++i) {
    Y[i] = (x_test[i%N] + v_test[i%N] + b_test[i%N] - E)*gamma_test[i%N]*rV + beta_test[i%N];
  }
  double t3 = omp_get_wtime();
  
  double t4 = omp_get_wtime();
  for (int i=0; i<vecSize; ++i) {
    Y[i] = (X[i] - E)*gamma_test[i%N]*rV + beta_test[i%N];
  }
  double t5 = omp_get_wtime();

  double l2 = 0.0;
  double t6 = omp_get_wtime();
  for (int i=0; i<vecSize; ++i) {
    double diff = Y[i] - Y[i%N];
    l2 += diff*diff;
  }
  double t7 = omp_get_wtime();
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
