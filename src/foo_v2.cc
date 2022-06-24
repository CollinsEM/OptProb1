#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#include <test_inputs.h>
#include <emc_utils.h>

// Version 0.2
//
// First attempt to implement loops in parallel using OpenMP
// directives.

int main(int argc, char ** argv) {
  int verbosity = 0;
  int vecSize = N;
  int numThreads = 1;
  int numCycles = 1;
  int parAvg = 0;
  // Parse command line arguments
  for (int i=1; i<argc; ++i) {
    // Specify the size of the input array
    if (!strcmp(argv[i], "-N") && (i+1) < argc) {
      vecSize = atoi( argv[++i] );
    }
    // Specify the number of threads
    else if (!strcmp(argv[i], "-nt") && (i+1) < argc) {
      numThreads = atoi( argv[++i] );
    }
    // Specify the number of cycles to run to average timings
    else if (!strcmp(argv[i], "-ncyc") && (i+1) < argc) {
      numCycles = atoi( argv[++i] );
    }
    // Verbosity can be increased by repeating '+v'
    else if (!strcmp(argv[i], "+v")) {
      verbosity++;
    }
    // Verbosity can be decreased by repeating '-v'
    else if (!strcmp(argv[i], "-v")) {
      verbosity--;
    }
    // Specify which averaging branch to execute (serial or parallel)
    else if (!strcmp(argv[i], "-pavg")) {
      parAvg = 1;
    }
  }
  
  if (verbosity > 0) printf("verbosity:  %d\n", verbosity);
  // Report input vector size
  if (verbosity > 0) printf("vecSize:    %d\n", vecSize);
  // Report number of threads
  if (verbosity > 0) printf("numThreads: %d\n", numThreads);
  omp_set_num_threads(numThreads);
  
  const real eps = 1e-5;
  
  // Array for recording time-stamps
  int NT = 5;
  double t[20], dt[10] = { 0.0 };

  real * X = new real[vecSize];
  real * Y = new real[vecSize];
  
  for (int n=0; n<numCycles; ++n) {
    int T=0;
    // NOTE: I'm splitting out the following steps of caching X and
    // computing avg & var so that it's easier to see any latency due to
    // the initial load of data values.
  
    // Cache the input vector x' = x + v + b
    t[T++] = omp_get_wtime();
#pragma omp parallel for
    for (int i=0; i<vecSize; ++i) {
      X[i] = x_test[i%N] + v_test[i%N] + b_test[i%N];
    }
    t[T++] = omp_get_wtime();
    if (verbosity > 0 && n==0) printf("%15.10f \t Time to compute x' = x + v + b.\n", t[T-1]-t[T-2]);

    real avg, var;
    if (!parAvg) {
      // This is a running-average calculator that I wrote a while
      // back. I typically use it when reading in a lot of data from a
      // file or a stream without having to store every value. It also
      // uses a variation on Kahan's summation method to reduce
      // truncation errors which accumulate when averaging very large
      // lists of numbers (i.e. when the sum begins to diverge from the
      // data values by more than a few orders of magnitude).
      //
      // NOTE: This algorithm has not yet been parallelized, so we'll
      // just handle it serially for now.
      OnlineAverage<real> xAvg;
      t[T++] = omp_get_wtime();
      for (int i=0; i<vecSize; ++i) {
        xAvg += X[i];
      }
      t[T++] = omp_get_wtime();
      avg = xAvg.mean();
      var = xAvg.variance();
      if (verbosity > 0 && n==0) {
        printf("%15.10f \t Time to compute mean and variance (serial).\n", t[T-1]-t[T-2]);
      }
    }
    else {
      // Compute the mean
      real sum = 0.0;
      t[T++] = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
      for (int i=0; i<vecSize; ++i) {
        sum += X[i];
      }
      if (verbosity > 2 && n == 0) printf("sum: %f\n", sum);
      avg = sum / vecSize;
      
      // Compute the variance
      real d, dSq = 0.0;
#pragma omp parallel for reduction(+ : dSq)
      for (int i=0; i<vecSize; ++i) {
        d = X[i] - avg;
        dSq += d*d;
      }
      var = dSq/(vecSize-1);
      t[T++] = omp_get_wtime();
      
      if (verbosity > 1 && n==0) {
        printf("OpenMP:\n");
        printf("sum: %f\n", sum);
        printf("avg: %f\n", avg);
        printf("var: %f\n", var);
        // Verify OpenMP parallel reduce
        sum = 0.0;
        for (int i=0; i<vecSize; ++i) {
          sum += X[i];
        }
        avg = sum / vecSize;
        
        dSq = 0.0;
        for (int i=0; i<vecSize; ++i) {
          d = X[i] - avg;
          dSq += d*d;
        }
        var = dSq/(vecSize-1);
        
        printf("Serial:\n");
        printf("sum: %f\n", sum);
        printf("avg: %f\n", avg);
        printf("var: %f\n", var);
      }
      if (verbosity > 0 && n==0) {
        printf("%15.10f \t Time to compute mean and variance (OpenMP).\n", t[T-1]-t[T-2]);
      }
    }
    if (verbosity > 1 && n==0) {
      printf("x-avg: %f\n", avg);
      printf("x-var: %f\n", var);
    }
  
    // Save values for computing Y
    const real E  = avg;
    const real rV = 1.0/(sqrt(var) + eps);
    
//     t[T++] = omp_get_wtime();
// #pragma omp parallel for
//     for (int i=0; i<vecSize; ++i) {
//       Y[i] = (x_test[i%N] + v_test[i%N] + b_test[i%N] - E)*gamma_test[i%N]*rV + beta_test[i%N];
//     }
//     t[T++] = omp_get_wtime();
//     if (verbosity > 0 && n==0) {
//       printf("%15.10f \t Time to compute Y from x, v, and b.\n", t[T-1]-t[T-2]);
//     }
  
    t[T++] = omp_get_wtime();
#pragma omp parallel for
    for (int i=0; i<vecSize; ++i) {
      Y[i] = (X[i] - E)*gamma_test[i%N]*rV + beta_test[i%N];
    }
    t[T++] = omp_get_wtime();
    if (verbosity > 0 && n==0) {
      printf("%15.10f \t Time to compute Y from x'.\n", t[T-1]-t[T-2]);
    }
    
    real l2 = 0.0;
    t[T++] = omp_get_wtime();
#pragma omp parallel for reduction(+ : l2)
    for (int i=0; i<vecSize; ++i) {
      real diff = Y[i] - Y_test[i%N];
      l2 += diff*diff;
    }
    const real L2 = sqrt(l2/vecSize);
    t[T++] = omp_get_wtime();
    if (verbosity > 0 && n==0) {
      printf("%15.10f \t Time to compute L2 error for Y.\n", t[T-1]-t[T-2]);
    }
    if (n==0) {
      printf("Sum of squared errors in the output vector Y: %10.6g.\n", L2);
    }
    
    NT = T/2;
    for (int i=0; i<NT; ++i) {
      dt[i] += t[2*i+1] - t[2*i];
    }
  }
  
  delete [] X;
  delete [] Y;

  double dT = 0.0;
  for (int i=0; i<NT; ++i) {
    dt[i] /= numCycles;
    dT += dt[i];
  }
  
  printf("Execution times (averaged over %d cycles)\n", numCycles);
  for (int i=0; i<NT; ++i) {
    printf("%10.4g \t %6.2f%%\n", dt[i], 100*dt[i]/dT);
  }
  printf("------------------\n");
  printf("%10.4g \t %6.2f%% \n", dT, 100.0);
  printf("\n");
  
  if (verbosity > 1) {
    printf("t[%2d]: %20.10f\n", 0, t[0]);
    for (int i=1; i<2*NT; ++i) {
      printf("t[%2d]: %20.10f  %15.6g  %15.6g\n", i, t[i], t[i]-t[i-1], t[i]-t[0]);
    }
  }
  
  return 0;
}
