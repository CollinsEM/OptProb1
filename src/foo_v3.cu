#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <omp.h>

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#else
#error CUDA is not available.
#endif

#include <test_inputs.h>
#include <emc_utils.h>

// Version 0.3
//
// First attempt at using AVX intrinsics
//

typedef float real;

__global__
void computeX(real * X, const real * x, const real * v, const real * b, int N, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    X[idx] = x[idx%N] + v[idx%N] + b[idx%N];
  }
}

__global__
void duplicate(real * X, const real * x, int N, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    X[idx] = x[idx%N];
  }
}

__global__
void computeDelta(real * delta, real * X, real E, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    delta[idx] = X[idx%N] - E;
  }
}

__global__
void computeDeltaSquared(real * dSq, real * X, real E, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    const real delta = X[idx] - E;
    dSq[idx] = delta*delta;
  }
}

__global__
void computeY(real * Y, const real * X, const real * G, const real * B, real xBar, real rVar, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    Y[idx] = (X[idx] - xBar)*rVar*G[idx] + B[idx];
  }
}

//--------------------------------------------------------------------
/// Compute x' = x + v + b
//--------------------------------------------------------------------
typedef thrust::tuple<real,real,real> VEC3;
//--------------------------------------------------------------------
struct ComputeX : public thrust::unary_function<VEC3, real> {
public:
  __device__
  real operator()(const VEC3 & xvb) const {
    return (thrust::get<0>(xvb) + thrust::get<1>(xvb) + thrust::get<2>(xvb));
  }
};
//--------------------------------------------------------------------
__constant__ real XBAR;
//--------------------------------------------------------------------
struct Delta : public thrust::unary_function<real, real> {
public:
  Delta(const real xb) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(real));
  }
  __device__
  real operator()(const real & x) const {
    return x - XBAR;
  }
};
//--------------------------------------------------------------------
struct DeltaSquared {
public:
  DeltaSquared(const real xb) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(real));
  }
  __device__
  real operator()(const real & x) const {
    const real delta = x - XBAR;
    return delta*delta;
  }
  __device__
  real operator()(const real & x1, const real x2) const {
    const real delta = x1 - x2;
    return delta*delta;
  }
};

//--------------------------------------------------------------------
/// Compute Y = (x' - xbar)/sqrt(var
//--------------------------------------------------------------------
__constant__ real XVAR;
__constant__ real rVAR;
//--------------------------------------------------------------------
struct ComputeY : public thrust::unary_function<real, real> {
public:
  ComputeY(const real xb, const real xv) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(real));
    cudaMemcpyToSymbol(XVAR, &xv, sizeof(real));
    const real rV = 1.0/(sqrt(xv) + 1e-5);
    cudaMemcpyToSymbol(rVAR, &rV, sizeof(real));
  }
  __device__
  real operator()(const VEC3 & v3) const {
    return (thrust::get<0>(v3) - XBAR)*rVAR*thrust::get<1>(v3) + thrust::get<2>(v3);
  }
};

int main(int argc, char ** argv) {
  int verbosity = 0;
  int vecSize = N;
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
    // Verbosity can be increased by repeating '+v'
    else if (!strcmp(argv[i], "+v")) {
      verbosity++;
    }
    // Verbosity can be decreased by repeating '-v'
    else if (!strcmp(argv[i], "-v")) {
      verbosity--;
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
  
  // NOTE: I'm splitting out the following steps of caching X and
  // computing avg & var so that it's easier to see any latency due to
  // the initial load of data values.
  
  const dim3 G((vecSize+1023)/1024);
  const dim3 B(min(1024,vecSize));
  // Load and cache the input vector x' = x + v + b
  thrust::device_vector<real> dx(N), dy(N), dv(N), db(N), dgamma(N), dbeta(N);
  real * px = thrust::raw_pointer_cast(dx.data());
  real * py = thrust::raw_pointer_cast(dy.data());
  real * pv = thrust::raw_pointer_cast(dv.data());
  real * pb = thrust::raw_pointer_cast(db.data());
  real * pgamma = thrust::raw_pointer_cast(dgamma.data());
  real * pbeta = thrust::raw_pointer_cast(dbeta.data());
  thrust::device_vector<real> dX(vecSize), dY(vecSize), dGamma(vecSize), dBeta(vecSize);
  real * pX = thrust::raw_pointer_cast(dX.data());
  real * pY = thrust::raw_pointer_cast(dY.data());
  real * pGamma = thrust::raw_pointer_cast(dGamma.data());
  real * pBeta = thrust::raw_pointer_cast(dBeta.data());
  // Copy over the first N data points
  thrust::copy(x_test, x_test+N, dx.begin());
  thrust::copy(Y_test, Y_test+N, dy.begin());
  thrust::copy(v_test, v_test+N, dv.begin());
  thrust::copy(b_test, b_test+N, db.begin());
  thrust::copy(gamma_test, gamma_test+N, dgamma.begin());
  thrust::copy(beta_test,  beta_test+N,  dbeta.begin());
  for (int n=0; n<numCycles; ++n) {
    int T=0;
    t[T++] = omp_get_wtime();
    computeX<<<G, B>>>(pX, px, pv, pb, N, vecSize);
    duplicate<<<G, B>>>(pGamma, pgamma, N, vecSize);
    duplicate<<<G, B>>>(pBeta, pbeta, N, vecSize);
    t[T++] = omp_get_wtime();
  
    t[T++] = omp_get_wtime();
    const real xSum = thrust::reduce(dX.begin(), dX.end());
    const real xBar = xSum/vecSize;
    t[T++] = omp_get_wtime();
    if (verbosity > 1 && n == 0) printf("xSum: %10.6g, xBar: %10.6g\n", xSum, xBar);

    t[T++] = omp_get_wtime();
    DeltaSquared deltaSq(xBar);
    const real sumDeltaSq = thrust::transform_reduce(dX.begin(), dX.end(), deltaSq, 0.0, thrust::plus<real>());
    const real xVar = sumDeltaSq/(vecSize-1);
    const real rVar = 1.0/(sqrt(xVar) + eps);
    t[T++] = omp_get_wtime();
    if (verbosity > 1 && n == 0) printf("sumDeltaSq: %10.6g, xVar: %10.6g, rVar: %10.6g\n", sumDeltaSq, xVar, rVar);

    // thrust::transform(make_zip_iterator(make_tuple(dX.begin(),dGamma.begin(),dBeta.begin())),
    //                   make_zip_iterator(make_tuple(dX.end(),  dGamma.end(),  dBeta.end())),
    //                   dY.begin());
    t[T++] = omp_get_wtime();
    computeY<<<G, B>>>(pY, pX, pGamma, pBeta, xBar, rVar, vecSize);
    t[T++] = omp_get_wtime();
    
    // t[T++] = omp_get_wtime();
    // computeL2<<<G, B>>>(pY, pX, pGamma, pBeta, xBar, rVar, vecSize);
    // t[T++] = omp_get_wtime();
    
    // real t6 = omp_get_wtime();
    // for (int i=0; i<vecSize; ++i) {
    //   real diff = Y[i] - Y[i%N];
    //   l2 += diff*diff;
    // }
    // real t7 = omp_get_wtime();
    // printf("Sum of squared errors in the output vector Y: %g\n", sqrt(l2/vecSize));

    NT = T/2;
    for (int i=0; i<NT; ++i) {
      dt[i] += t[2*i+1] - t[2*i];
    }
    
  }

  // delete [] X;
  // delete [] Y;
  
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
  
  printf("%10.4g \t %6.2f%% \t Time to compute mean and variance (and X)\n", dt[0], 100*dt[0]/dT);
  // printf("%10.4g \t %6.2f%% \t Time to compute output vector (without X)\n", t3-t2, 100*(t3-t2)/(t7-t0));
  // printf("%10.4g \t %6.2f%% \t Time to compute output vector (with X)\n", t5-t4, 100*(t5-t4)/(t7-t0));
  // printf("%10.4g \t %6.2f%% \t Time to compute L2 error norm\n", t7-t6, 100*(t7-t6)/(t7-t0));
  // printf("------------------\n");
  // printf("%10.4g \t %6.2f%% \t Total execution time\n", t7-t0, 100.0);
  // printf("\n");
  return 0;
}
