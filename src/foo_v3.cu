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
// CUDA accelerated implementation
//
__global__
void computeX(float * X, const float * x, const float * v, const float * b, int N, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    X[idx] = x[idx%N] + v[idx%N] + b[idx%N];
  }
}

__global__
void duplicate(float * X, const float * x, int N, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    X[idx] = x[idx%N];
  }
}

__global__
void computeDelta(float * delta, float * X, float E, int sz) {
  const int idx = threadIdx.x + blockIdx.x*blockDim.x;
  if (idx < sz) {
    delta[idx] = X[idx%N] - E;
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

#define FULL_MASK 0xffffffff
__device__
float reduce_sum(const float x, const int sz) {
  __shared__ float data[2048];
  const int gid = threadIdx.x + blockIdx.x*blockDim.x; // global thread ID
  const int tid  = gid & 0xFF; // gid%32 : warp-local thread ID
  const int wid  = gid >> 5;   // gid/32 : warp ID
  // Store partial sum value of x to be reduced.
  float sum = x;
  int N = sz;
  while (N>0) {
    unsigned mask = __ballot_sync(FULL_MASK, gid<N);
    // Perform a local reduction over the warp
    for (int offset=16; offset>0; offset/=2) {
      sum += __shfl_down_sync(mask, sum, offset);
      // __syncwarp();
    }
    // Store the warp-local partial sums in block-shared memory
    if (tid == 0) data[wid] = sum;
    __syncthreads();
    // Reduce the number of active threads
    N >>= 5; // N /= 32;
    // Update the thread-local values to be reduced.
    sum = (gid < N ? data[gid] : 0.0);
  }
  return data[0];
}

__global__
void fusedKernel(float *Y, const float *x, const float *v, const float *b,
                 const float *gamma, const float *beta, const int sz) {
  const int gid = threadIdx.x + blockIdx.x*blockDim.x; // global thread ID
  const int tid  = gid & 0xFF; // gid%32 : warp-local thread ID
  const int wid  = gid >> 5;   // gid/32 : warp ID
  // Compute the thread-local value of x' = (x + v + b)
  float xp = (idx<sz ? x[idx] + v[idx] + b[idx] : 0.0);
  // Store partial sum value of xp/sz to be reduced.
  float avg = xp/sz;
  int N = sz;
  while (N > 0) {
    unsigned mask = __ballot_sync(FULL_MASK, gid<N);
    // Perform a local reduction over the warp
    for (int offset=16; offset>0; offset/=2) {
      avg += __shfl_down_sync(mask, avg, offset);
      // __syncwarp();
    }
    // Store the warp-local partial sums in block-shared memory
    if (tid == 0) data[wid] = avg;
    __syncthreads();
    // Reduce the number of active threads
    N >>= 5; // N /= 32;
    // Update the thread-local values to be reduced.
    avg = (gid < N ? data[gid] : 0.0);
  }
  avg = data[0];
  float dX = xp - avg;
  float var = dX*dX/(sz-1);
  N = sz;
  while (N > 0) {
    unsigned mask = __ballot_sync(FULL_MASK, gid<N);
    // Perform a local reduction over the warp
    for (int offset=16; offset>0; offset/=2) {
      var += __shfl_down_sync(mask, var, offset);
      // __syncwarp();
    }
    // Store the warp-local partial sums in block-shared memory
    if (tid == 0) data[wid] = var;
    __syncthreads();
    // Reduce the number of active threads
    N >>= 5; // N /= 32;
    // Update the thread-local values to be reduced.
    var = (gid < N ? data[gid] : 0.0);
  }
  float rVar = 1.0/sqrt
  Y[idx] = (x[idx] + v[idx] + b[idx] - xBar)*rVar*G[idx] + B[idx];
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
struct DeltaSquared {
public:
  DeltaSquared(const float xb) {
    cudaMemcpyToSymbol(XBAR, &xb, sizeof(float));
  }
  __device__
  float operator()(const float & x) const {
    const float delta = x - XBAR;
    return delta*delta;
  }
  __device__
  float operator()(const float & x1, const float x2) const {
    const float delta = x1 - x2;
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
  
  const float eps = 1e-5;
  
  // Array for recording time-stamps
  int NT = 5;
  double t[20], dt[10] = { 0.0 };
  
  // NOTE: I'm splitting out the following steps of caching X and
  // computing avg & var so that it's easier to see any latency due to
  // the initial load of data values.
  
  const dim3 G((vecSize+1023)/1024);
  const dim3 B(min(1024,vecSize));
  // Load and cache the input vector x' = x + v + b
  thrust::device_vector<float> dx(N), dy(N), dv(N), db(N), dgamma(N), dbeta(N);
  float * px = thrust::raw_pointer_cast(dx.data());
  float * py = thrust::raw_pointer_cast(dy.data());
  float * pv = thrust::raw_pointer_cast(dv.data());
  float * pb = thrust::raw_pointer_cast(db.data());
  float * pgamma = thrust::raw_pointer_cast(dgamma.data());
  float * pbeta = thrust::raw_pointer_cast(dbeta.data());
  thrust::device_vector<float> dX(vecSize), dY(vecSize), dGamma(vecSize), dBeta(vecSize);
  float * pX = thrust::raw_pointer_cast(dX.data());
  float * pY = thrust::raw_pointer_cast(dY.data());
  float * pGamma = thrust::raw_pointer_cast(dGamma.data());
  float * pBeta = thrust::raw_pointer_cast(dBeta.data());
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
    const float xSum = thrust::reduce(dX.begin(), dX.end());
    const float xBar = xSum/vecSize;
    t[T++] = omp_get_wtime();
    if (verbosity > 1 && n == 0) printf("xSum: %10.6g, xBar: %10.6g\n", xSum, xBar);

    t[T++] = omp_get_wtime();
    DeltaSquared deltaSq(xBar);
    const float sumDeltaSq = thrust::transform_reduce(dX.begin(), dX.end(), deltaSq, 0.0, thrust::plus<float>());
    const float xVar = sumDeltaSq/(vecSize-1);
    const float rVar = 1.0/(sqrt(xVar) + eps);
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
    
    // double t6 = omp_get_wtime();
    // for (int i=0; i<vecSize; ++i) {
    //   float diff = Y[i] - Y[i%N];
    //   l2 += diff*diff;
    // }
    // double t7 = omp_get_wtime();
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
