#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#include <test_inputs.h>
#include <emc_utils.h>

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
  OnlineAverage<double> xAvg;
  double t0 = omp_get_wtime();
  double * X = new double[N];
  for (int i=0; i<N; ++i) {
    X[i] = x_test[i%64] + v_test[i%64] + b_test[i%64];
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
  double * Y = new double[N];
  double t2 = omp_get_wtime();
  for (int i=0; i<N; ++i) {
    Y[i] = (x_test[i%64] + v_test[i%64] + b_test[i%64] - E)*gamma_test[i%64]*rV + beta_test[i%64];
  }
  double t3 = omp_get_wtime();
  
  double t4 = omp_get_wtime();
  for (int i=0; i<N; ++i) {
    Y[i] = (X[i] - E)*gamma_test[i%64]*rV + beta_test[i%64];
  }
  double t5 = omp_get_wtime();

  double l2 = 0.0;
  double t6 = omp_get_wtime();
  for (int i=0; i<N; ++i) {
    double diff = Y[i] - Y[i%64];
    l2 += diff*diff;
  }
  double t7 = omp_get_wtime();
  printf("Sum of squared errors in the output vector Y.\n");
  printf("L2: %g\n", sqrt(l2/N));

  delete [] X;
  delete [] Y;

  printf("t1-t0 = %10.6g \tTime to compute mean and variance (and X)\n", t1-t0);
  printf("t3-t2 = %10.6g \tTime to compute output vector (without X)\n", t3-t2);
  printf("t5-t4 = %10.6g \tTime to compute output vector (with X)\n", t5-t4);
  printf("t7-t6 = %10.6g \tTime to compute L2 error norm\n", t5-t4);
  printf("------------------\n");
  printf("t7-t0 = %10.6g Total execution time\n", t5-t0);
  printf("\n");
  return 0;
}
