#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
using std::vector;

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

typedef float real;

int main(int argc, char ** argv) {
  bool verbose = false;
  int vecSize = N;
  int numCycles = 1;
  double t[20], dt[10] = { 0.0 };
  // Parse command line arguments
  for (int i=1; i<argc; ++i) {
    // Specify the size of the input array
    if (!strcmp(argv[i], "-N") && (i+1) < argc) {
      vecSize = atoi( argv[++i] );
    }
    // Specify the number of cycles to run to average timings
    else if (!strcmp(argv[i], "-ncyc") && (i+1) < argc) {
      numCycles = atoi( argv[++i] );
    }
    // Parse single parameter switches
    else if (!strcmp(argv[i], "-v")) {
      verbose = true;
    }
  }
  // Report input vector size
  if (verbose) printf("vecSize: %d\n", vecSize);
  
  real * X = new real[vecSize];
  real * Y = new real[vecSize];
  
  for (int n=0; n<numCycles; ++n) {
    // This is a running average calculator that I use quite a lot when
    // reading in data from a file or a stream without having to store
    // every value.
    OnlineAverage<real> xAvg;
    t[0] = omp_get_wtime();
    for (int i=0; i<vecSize; ++i) {
      X[i] = x_test[i%N] + v_test[i%N] + b_test[i%N];
      xAvg += X[i];
    }
    t[1] = omp_get_wtime();

    const real eps = 1e-5;
    const real E = xAvg.mean();
    const real rV = 1.0/xAvg.variance() + eps;
    t[2] = omp_get_wtime();
    for (int i=0; i<vecSize; ++i) {
      Y[i] = (x_test[i%N] + v_test[i%N] + b_test[i%N] - E)*gamma_test[i%N]*rV + beta_test[i%N];
    }
    t[3] = omp_get_wtime();
    
    t[4] = omp_get_wtime();
    for (int i=0; i<vecSize; ++i) {
      Y[i] = (X[i] - E)*gamma_test[i%N]*rV + beta_test[i%N];
    }
    t[5] = omp_get_wtime();
    
    real l2 = 0.0;
    t[6] = omp_get_wtime();
    for (int i=0; i<vecSize; ++i) {
      real diff = Y[i] - Y[i%N];
      l2 += diff*diff;
    }
    t[7] = omp_get_wtime();
    if (verbose && n==0) {
      printf("Sum of squared errors in the output vector Y: %g\n",
             sqrt(l2/vecSize));
    }

    for (int i=0; i<4; ++i) {
      dt[i] += t[2*i+1] - t[2*i];
    }
    
  }
  
  delete [] X;
  delete [] Y;
  
  double dT = 0.0;
  for (int nt=0; nt<4; ++nt) {
    dt[nt] /= numCycles;
    dT += dt[nt];
  }
  
  printf("%10.4g \t %6.2f%% \t Avg. time to compute mean and variance (and X)\n", dt[0], 100*dt[0]/dT);
  printf("%10.4g \t %6.2f%% \t Avg. time to compute output vector (without X)\n", dt[1], 100*dt[1]/dT);
  printf("%10.4g \t %6.2f%% \t Avg. time to compute output vector (with X)\n", dt[2], 100*dt[2]/dT);
  printf("%10.4g \t %6.2f%% \t Avg. time to compute L2 error norm\n", dt[3], 100*dt[3]/dT);
  printf("------------------\n");
  printf("%10.4g \t %6.2f%% \t Avg. total execution time\n", dT, 100*dT/dT);
  printf("\n");
  return 0;
}
