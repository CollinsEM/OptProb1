#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <sstream>
#include <vector>

#include <omp.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#else
#error CUDA is not available.
#endif

#include <TestVec.h>
#include <emc_utils.h>

// Version 0.3
//
// CUDA accelerated implementation
//
// Parse command line arguments
//  * +v : More verbose output
//  * -v : Less verbose output
//  * -N <int> : Set size of data array
//  * -ncyc <int> : Set number of cycles to run
//
// Use OpenMP's high-resolution timing. Run multiple cycles to
// generate average timings.
__global__
void computeX(float * X, const float * x, const float * v, const float * b, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    X[idx] = x[idx] + v[idx] + b[idx];
  }
}

__global__
void computeDelta(float * delta, float * X, float E, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    delta[idx] = X[idx] - E;
  }
}

__global__
void computeDeltaSquared(float * dSq, float * X, float E, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    const float delta = X[idx] - E;
    dSq[idx] = delta*delta;
  }
}

__global__
void computeY(float * Y, const float * X, const float * G, const float * B, float xBar, float rVar, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    Y[idx] = (X[idx] - xBar)*rVar*G[idx] + B[idx];
  }
}

__global__
void computeSqErrors(float * dYSq, const float * Y, const float * y, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    const float delta = (Y[idx] - y[idx]);
    dYSq[idx] = delta*delta;
  }
}

//--------------------------------------------------------------------
/// Compute x' = x + v + b
//--------------------------------------------------------------------
typedef thrust::tuple<float,float,float> VEC3;
//--------------------------------------------------------------------
struct ComputeX : public thrust::unary_function<VEC3, float> {
public:
  __device__
  float operator()(const VEC3 & xvb) const {
    return (thrust::get<0>(xvb) + thrust::get<1>(xvb) + thrust::get<2>(xvb));
  }
};
//--------------------------------------------------------------------
__constant__ float XBAR;
//--------------------------------------------------------------------
struct Delta : public thrust::unary_function<float, float> {
public:
  Delta(const float xb) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(float));
  }
  __device__
  float operator()(const float & x) const {
    return x - XBAR;
  }
};
//--------------------------------------------------------------------
struct DeltaXSquared {
public:
  DeltaXSquared(const float xb) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(float));
  }
  __device__
  float operator()(const float & x) const {
    const float delta = x - XBAR;
    return delta*delta;
  }
};
//--------------------------------------------------------------------
struct DeltaYSquared {
  typedef thrust::tuple<float,float> VEC2;
public:
  __device__
  float operator()(const VEC2 & Y) const {
    const float delta = thrust::get<0>(Y) - thrust::get<1>(Y);
    return delta*delta;
  }
};

//--------------------------------------------------------------------
/// Compute Y = (x' - xbar)/sqrt(var
//--------------------------------------------------------------------
__constant__ float XVAR;
__constant__ float rVAR;
//--------------------------------------------------------------------
struct ComputeY : public thrust::unary_function<float, float> {
public:
  ComputeY(const float xb, const float xv) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(float));
    cudaMemcpyToSymbol(XVAR, &xv, sizeof(float));
    const float rV = 1.0/(sqrt(xv) + 1e-5);
    cudaMemcpyToSymbol(rVAR, &rV, sizeof(float));
  }
  __device__
  float operator()(const VEC3 & v3) const {
    return (thrust::get<0>(v3) - XBAR)*rVAR*thrust::get<1>(v3) + thrust::get<2>(v3);
  }
};

int main(int argc, char ** argv) {
  int verbose = 0;
  int vecSize = 64;
  // int numThreads = 1;
  int numCycles = 1;
  // Parse command line arguments
  for (int i=1; i<argc; ++i) {
    // Specify the size of the input array
    if (!strcmp(argv[i], "-N") && (i+1) < argc) {
      vecSize = atoi( argv[++i] );
    }
    // Specify the number of threads
    // else if (!strcmp(argv[i], "-nt") && (i+1) < argc) {
    //   numThreads = atoi( argv[++i] );
    // }
    // Specify the number of cycles to run to average timings
    else if (!strcmp(argv[i], "-ncyc") && (i+1) < argc) {
      numCycles = atoi( argv[++i] );
    }
    // Verbosity can be increased by repeating '+v'
    else if (!strcmp(argv[i], "+v")) {
      verbose++;
    }
    // Verbosity can be decreased by repeating '-v'
    else if (!strcmp(argv[i], "-v")) {
      verbose--;
    }
  }
  
  if (verbose > 0) printf("verbose:  %d\n", verbose);
  // Report input vector size
  if (verbose > 0) printf("vecSize:    %d\n", vecSize);
  // Report number of threads
  // if (verbose > 0) printf("numThreads: %d\n", numThreads);
  // omp_set_num_threads(numThreads);
  
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
  
  const dim3 G((vecSize+1023)/1024);
  const dim3 B(min(1024,vecSize));
  // Input vectors
  thrust::device_vector<float> dx(vecSize);
  thrust::device_vector<float> dv(vecSize);
  thrust::device_vector<float> db(vecSize);
  thrust::device_vector<float> dgamma(vecSize);
  thrust::device_vector<float> dbeta(vecSize);
  float * px = thrust::raw_pointer_cast(dx.data());
  float * pv = thrust::raw_pointer_cast(dv.data());
  float * pb = thrust::raw_pointer_cast(db.data());
  float * pgamma = thrust::raw_pointer_cast(dgamma.data());
  float * pbeta = thrust::raw_pointer_cast(dbeta.data());
  // Output vectors
  thrust::device_vector<float> dX(vecSize);
  thrust::device_vector<float> dY(vecSize);
  thrust::device_vector<float> dy(vecSize);
  float * pX = thrust::raw_pointer_cast(dX.data());
  float * pY = thrust::raw_pointer_cast(dY.data()); // gpu solution
  float * py = thrust::raw_pointer_cast(dy.data()); // cpu solution
  
  for (int n=0; n<numCycles; ++n) {
    int p(0), q(0);
    
    // Initialize the input vectors with random elements for their
    // respective distributions.
    x.init();
    v.init();
    b.init();
    gamma.init();
    beta.init();
    // Generate the solution vector (only if needed)
    if (verbose > 0) y.eval(x, v, b, gamma, beta);
    
    // Copy the input vectors to the device
    thrust::copy(x.data, x.data+vecSize, dx.begin());
    thrust::copy(v.data, v.data+vecSize, dv.begin());
    thrust::copy(b.data, b.data+vecSize, db.begin());
    thrust::copy(gamma.data, gamma.data+vecSize, dgamma.begin());
    thrust::copy(beta.data,  beta.data+vecSize,  dbeta.begin());
    // Copy the CPU solution vector to the device
    thrust::copy(y.data, y.data+vecSize, dy.begin());
    
    if (n==0) lbls.push_back("Compute X");
    t[q++] = omp_get_wtime();
    computeX<<<G, B>>>(pX, px, pv, pb, vecSize);
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];

    if (n==0) lbls.push_back("Compute X-bar");
    t[q++] = omp_get_wtime();
    const float xSum = thrust::reduce(dX.begin(), dX.end());
    const float xBar = xSum/vecSize;
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];
    if (verbose > 1 && n == 0) printf("xSum: %10.6g, xBar: %10.6g\n", xSum, xBar);

    if (n==0) lbls.push_back("Compute X-var");
    t[q++] = omp_get_wtime();
    DeltaXSquared deltaXSq(xBar);
    const float sumDeltaXSq = thrust::transform_reduce(dX.begin(), dX.end(), deltaXSq, 0.0, thrust::plus<float>());
    const float xVar = sumDeltaXSq/(vecSize-1);
    const float rVar = 1.0/(sqrt(xVar) + 1e-8);
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];
    if (verbose > 1 && n == 0) printf("sumDeltaXSq: %10.6g, xVar: %10.6g, rVar: %10.6g\n", sumDeltaXSq, xVar, rVar);

    if (n==0) lbls.push_back("Compute Y");
    t[q++] = omp_get_wtime();
    computeY<<<G, B>>>(pY, pX, pgamma, pbeta, xBar, rVar, vecSize);
    t[q++] = omp_get_wtime();
    dt[p++] += t[q-1] - t[q-2];

    if (verbose > 0) {
      if (n == 0) lbls.push_back("Compute L2-errors");
      t[q++] = omp_get_wtime();
      float sumDeltaYSq = thrust::transform_reduce( make_zip_iterator(make_tuple(dY.begin(), dy.begin())),
                                                    make_zip_iterator(make_tuple(dY.end(),   dy.end())),
                                                    DeltaYSquared(), 0.0f, thrust::plus<float>() );
      t[q++] = omp_get_wtime();
      dt[p++] += t[q-1] - t[q-2];
      if (n == 0) printf("sumDeltaYSq: %10.6g\n", sumDeltaYSq);
    }
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
  oss << "data/v3_" << vecSize << ".dat";
  fp = fopen(oss.str().c_str(), "w");
  fprintf(fp, "%12d", vecSize);
  for (int i=0; i<NT; ++i) {
    fprintf(fp, "%12g", dt[i].mean());
  }
  fclose(fp);
  
  return 0;
}
