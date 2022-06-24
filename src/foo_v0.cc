#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#include <test_inputs.h>
#include <emc_utils.h>

// Version 0.0.0
//
// Initial implementation.
//
//  * Parse command line arguments
//    * -v: Verbose output
//    * -N n: Data array size
//
//  * Compute mean and variance using two different methods to verify
//    correct implementation.
int main(int argc, char ** argv) {
  bool verbose = false;
  // Parse command line arguments
  for (int i=1; i<argc; ++i) {
    if (strcmp(argv[i], "-v") == 0) {
      verbose = true;
    }
    else if (strcmp(argv[i], "-N") == 0 && (i+1)<argc) {
      N = atoi(argv[++i]);
    }
  }
  if (verbose) {
    for(int i=0; i<argc; ++i) {
      printf("%s ", argv[i]);
    }
    printf("\n");
  }
  // Report input vector size
  if (verbose) printf("N: %d\n", N);
  
  // This is a running average calculator that I use quite a lot when
  // reading in data from a file or a stream without having to store
  // every value.
  OnlineAverage<float> xAvg;
  float t0 = omp_get_wtime();
  for (int i=0; i<N; ++i) {
    xAvg += x_test[i%64] + v_test[i%64] + b_test[i%64];
  }
  float t1 = omp_get_wtime();
  
  // Sanity check to make sure my library is correct
  float sum = 0.0;
  for (int i=0; i<N; ++i) {
    sum += x_test[i%64] + v_test[i%64] + b_test[i%64];
  }
  float avg = sum/N;
  float delta = 0.0;
  float tmp = 0.0;
  for (int i=0; i<N; ++i) {
    delta = x_test[i%64] + v_test[i%64] + b_test[i%64] - avg;
    tmp += delta*delta;
  }
  float var = tmp/(N-1);
  float dev = sqrt(var);
  
  printf("xAvg.count: %ld\n", xAvg.count());
  printf("xAvg.mean: %f %f\n", xAvg.mean(), avg);
  printf("xAvg.min: %f\n", xAvg.min());
  printf("xAvg.max: %f\n", xAvg.max());
  printf("xAvg.range: %f\n", xAvg.range());
  printf("xAvg.variance: %f %f\n", xAvg.variance(), var);
  printf("xAvg.stdDev: %f %f\n", xAvg.stdDev(), dev);

  printf("t1-t0 = %g\n", t1-t0);
  
  printf("\n");
  
  return 0;
}
