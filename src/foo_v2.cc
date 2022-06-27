#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

#include <omp.h>

#include <TestVec.h>
#include <emc_utils.h>

// Version 0.2
//
// First attempt to implement loops in parallel using OpenMP
// directives.
//
// Parse command line arguments
//  * +v : More verbose output
//  * -v : Less verbose output
//  * -N <int> : Set size of data array
//  * -ncyc <int> : Set number of cycles to run
//  * -nt <int> : Set number of threads
//
// Use OpenMP's high-resolution timing. Run multiple cycles to
// generate average timings.
int main(int argc, char ** argv) {
  int verbose = 0;
  int vecSize = 64;
  int numThreads = 1;
  int numCycles = 1;
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
    // Verbose can be increased by repeating '+v'
    else if (!strcmp(argv[i], "+v")) {
      verbose++;
    }
    // Verbose can be decreased by repeating '-v'
    else if (!strcmp(argv[i], "-v")) {
      verbose--;
    }
  }
  if (verbose > 0) printf("verbose:  %d\n", verbose);
  // Report input vector size
  if (verbose > 0) printf("vecSize:    %d\n", vecSize);
  // Report number of threads
  if (verbose > 0) printf("numThreads: %d\n", numThreads);
  omp_set_num_threads(numThreads);
  
  int nt;
  // Time stamps
  double t[32];
  // Keep a running average of the elapsed times
  OnlineAverage<double> dt[16];
  // Array of output strings
  std::vector<std::string> lbls;

  // Generate random test vectors
  TestVec<float> x(vecSize, 0.0, 10.0);
  TestVec<float> v(vecSize, 0.0, 1.0);
  TestVec<float> b(vecSize, 2.5, 1.5);
  TestVec<float> gamma(vecSize, 0.0, 1.0);
  TestVec<float> beta(vecSize, 0.5, 0.5);
  // Solution vector
  SolnVec<float> y(vecSize);
  float sumX, avgX, sumDX, varX, devX, rVar, dY, sumDY;
  Vec<float> X(vecSize), dX(vecSize), Y(vecSize);
  OnlineAverage<float> dXSq, dYSq;
  for (int n=0; n<numCycles; ++n) {
    int p(0), q(0);

    // Initialize the input vectors with random elements for their
    // respective distributions.
    x.init();
    v.init();
    b.init();
    gamma.init();
    beta.init();
    // Generate the solution vector
    y.eval(x, v, b, gamma, beta);

    // NOTE: I'm splitting out the following steps of caching X and
    // computing avg & var so that it's easier to see any latency due to
    // the initial load of data values.
  
    if (n==0) lbls.push_back("Compute X, xAvg");
    t[q++] = omp_get_wtime();
    // Compute the sum of all x' values
    sumX = 0.0;
#pragma omp parallel for
    for (int i=0; i<vecSize; ++i) {
      X[i] = x[i] + v[i] + b[i];
      sumX += X[i];
    }
    // Compute the average of x'
    avgX = sumX/vecSize;
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];

    if (n==0) lbls.push_back("Compute dX, xVar");
    t[q++] = omp_get_wtime();
    // Compute the variance of x'
    sumDX = 0.0;
#pragma omp parallel for reduction(+ : sumDX)
    for (int i=0; i<vecSize; ++i) {
      dX[i] = X[i] - avgX;
      sumDX += dX[i]*dX[i];
    }
    varX = sumDX/(vecSize-1);
    rVar = 1.0/(sqrt(varX)+1e-8);
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];
    dXSq += sumDX;

    if (n==0) lbls.push_back("Compute Y");
    t[q++] = omp_get_wtime();
#pragma omp parallel for
    for (int i=0; i<vecSize; ++i) {
      Y[i] = dX[i]*gamma[i]*rVar + beta[i];
    }
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];
    
    if (n==0) lbls.push_back("Compute Errors");
    t[q++] = omp_get_wtime();
    // Compute errors in Y
    sumDY = 0.0;
#pragma omp parallel for reduction(+ : sumDY)
    for (int i=0; i<vecSize; ++i) {
      dY = Y[i] - y[i];
      sumDY += dY*dY;
    }
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];
    dYSq += sumDY;
  }

  const int NT = lbls.size();
  double DT = 0.0;
  for (int i=0; i<NT; ++i) {
    DT += dt[i].mean();
  }
  if (verbose > 0) {
    for (int i=0; i<NT; ++i) {
      printf("%12g %12g %6.2f%% %s\n",
             dt[i].mean(), dt[i].stdDev(),
             100*dt[i].mean()/DT, lbls[i].c_str());
    }
    printf("------------------ ------------------\n");
    printf("%12g %12s %6.2f%% \t Avg. total execution time\n", DT, "", 100.0);
    printf("\n");
  }
  else {
    printf("%12d", vecSize);
    for (int i=0; i<NT; ++i) {
      printf("%12g", dt[i].mean());
    }
    printf("%12g\n", DT);
  }
  
  FILE * fp = NULL;
  std::ostringstream oss;
  oss << "v2_nt" << numThreads << "_" << vecSize << ".dat";
  fp = fopen(oss.str().c_str(), "w");
  fprintf(fp, "%12d", vecSize);
  for (int i=0; i<NT; ++i) {
    fprintf(fp, "%12g", dt[i].mean());
  }
  fclose(fp);
  
  return 0;
}
