#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include <omp.h>

#include <TestVec.h>
#include <emc_utils.h>

// Version 0.0
//
// Initial build. Test some utility data structures.
//
//  * Parse command line arguments
//    * +v : More verbose output
//    * -v : Less verbose output
//    * -N <int> : Set size of data array
//
//  * Compute mean and variance using two different methods to verify
//    correct implementation.
//
//  * Generate random input vectors to the correct lengths and with
//    normal distributions. Then compute the resulting output vector.
int main(int argc, char ** argv) {
  int verbose = 0;
  int vecSize = 64;
  // Parse command line arguments
  for (int i=1; i<argc; ++i) {
    // Specify verbosity level
    if ( !strcmp(argv[i], "+v") ) {
      verbose++;
    }
    else if ( !strcmp(argv[i], "-v") ) {
      verbose--;
    }
    // Specify the size of the input array
    else if ( !strcmp(argv[i], "-N") && (i+1)<argc ) {
      vecSize = atoi(argv[++i]);
    }
  }
  // Report input vector size
  if (verbose>0) printf("vecSize: %d\n", vecSize);

  // Random test vectors
  TestVec<float> x(vecSize, 0.0, 10.0);
  TestVec<float> v(vecSize, 0.0, 1.0);
  TestVec<float> b(vecSize, 2.5, 1.5);
  TestVec<float> gamma(vecSize, 0.0, 1.0);
  TestVec<float> beta(vecSize, 0.5, 0.5);
  // Solution vector
  SolnVec<float> y(vecSize);
  // Initialize the input vectors with random elements for their
  // respective distributions.
  x.init();
  v.init();
  b.init();
  gamma.init();
  beta.init();
  // Generate the solution vector
  y.eval(x, v, b, gamma, beta);
    
  // Sanity check to make sure the OnlineAverage in y is correct.
  // NOTE: This version makes two passes through the data.
  Vec<float> X(vecSize); // cache values for x' = (x + v + b)
  // Compute the sum of all x' values
  float sum = 0.0;
  for (int i=0; i<vecSize; ++i) {
    X[i] = x[i] + v[i] + b[i];
    sum += X[i];
  }
  // Compute the average of x'
  float avg = sum/vecSize;
  // Compute the variance of x'
  float delta = 0.0;
  float tmp = 0.0;
  for (int i=0; i<vecSize; ++i) {
    delta = X[i] - avg;
    tmp += delta*delta;
  }
  float var = tmp/(vecSize-1);
  float dev = sqrt(var);
  float rVar = 1.0/(sqrt(var)+1e-8);

  // Compute output vector from the above data and compare against
  // SolnVec
  Vec<float> Y(vecSize);
  OnlineAverage<float> dXSq, dYSq;
  for (int i=0; i<vecSize; ++i) {
    Y[i] = (X[i] - avg)*rVar*gamma[i] + beta[i];
    const double dX(X[i] - y.X[i]);
    const double dY(Y[i] - y[i]);
    dXSq += (dX*dX);
    dYSq += (dY*dY);
    if (verbose > 2) {
      printf("%12f %12f %12f", y.X[i], X[i], dX);
      printf("%12f %12f %12f", y[i], Y[i], dY);
      printf("\n");
    }
  }

  printf("          Online-Stats Sanity-Check Error\n");
  printf("count:    %12ld %12d\n", y.xStats.count(), vecSize);
  printf("mean:     %12f %12f %g\n", y.xStats.mean(), avg, fabs((y.xStats.mean()-avg)/avg));
  printf("variance: %12f %12f %g\n", y.xStats.variance(), var, fabs((y.xStats.variance()-var)/var));
  printf("stdDev:   %12f %12f %g\n", y.xStats.stdDev(), dev, fabs((y.xStats.stdDev()-dev)/dev)); 
  printf("dXSq:     %12f %12f\n", dXSq.mean(), dXSq.stdDev());
  printf("dYSq:     %12f %12f\n", dYSq.mean(), dYSq.stdDev());
 printf("\n");
  
  return 0;
}
