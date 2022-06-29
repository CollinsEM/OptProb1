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
//

// The GPU has a maximum limit of 1024 threads per block. So, each
// block can have up to 32 warps of 32 threads.
//
// A reduction within a block can be accomplished with two cycles of
// warp reductions.
//
// Each warp reduction can be accomplished in five iterations of
// warp-shuffle operations.
//
// The total number of grid blocks will be (sz+1023)/1024.

//--------------------------------------------------------------------
// 32-bit mask for threads in one warp
#define MASK32 0xffffffff
//--------------------------------------------------------------------
// Run this kernel with block size of min(1024,N) and grid size of 1.
//--------------------------------------------------------------------
__global__
void block1D_reduce_sum(float * sumX, const float * X, int N) {
  __shared__ float data[32];
  // Do a partial sum of strided array elements in serial to bring down
  // the number of values to fit into a single block.
  float sum = 0.0;
  for (int idx=threadIdx.x; idx<N; idx+=blockDim.x) {
    sum += X[idx];
  }
  // Do a parallel reduction on these partial sums. Start with
  // warp-level reductions.
  for (int s=16; s>0; s/=2) {
    sum += __shfl_down_sync(MASK32, sum, s);
  }
  // Communicate partial results in each warp using block-shared memory
  if ((threadIdx.x % 32) == 0) data[threadIdx.x / 32] = sum;
  __syncthreads();
  sum = data[threadIdx.x % 32];
  // Do a final parallel reduction on the remaining values.
  for (int s=16; s>0; s/=2) {
    sum += __shfl_down_sync(MASK32, sum, s);
  }
  if (threadIdx.x == 0) {
    *sumX = sum;
  }
}

//--------------------------------------------------------------------
// Run this kernel with block size of min(1024,N) and grid size of 1.
//--------------------------------------------------------------------
__global__
void computeY(float *Y, const float *x, const float *v, const float *b,
              const float *gamma, const float *beta, const int N) {
  __shared__ float data[32];
  // Compute the mean for the x' values
  float sum = 0.0;
  for (int idx=threadIdx.x; idx<N; idx+=blockDim.x) {
    sum += x[idx] + v[idx] + b[idx];
  }
  for (int s=16; s>0; s/=2) {
    sum += __shfl_down_sync(MASK32, sum, s);
  }
  if ((threadIdx.x % 32) == 0) data[threadIdx.x / 32] = sum;
  
  __syncthreads();
  
  sum = data[threadIdx.x % 32];
  for (int s=16; s>0; s/=2) {
    sum += __shfl_down_sync(MASK32, sum, s);
  }
  if (threadIdx.x == 0) {
    data[0] = sum;
    // printf("sum: %f\n", sum);
    // printf("avg: %f\n", sum/N);
  }
  
  __syncthreads();
  
  // Distribute the average x' value to all threads
  float xBar = data[0]/N;

  // Compute the variance for the x' values
  float dX, dXSq = 0.0;
  for (int idx=threadIdx.x; idx<N; idx+=blockDim.x) {
    dX = x[idx] + v[idx] + b[idx] - xBar;
    dXSq += dX*dX;
  }
  for (int s=16; s>0; s/=2) {
    dXSq += __shfl_down_sync(MASK32, dXSq, s);
  }
  if ((threadIdx.x % 32) == 0) data[threadIdx.x / 32] = sum;
  
  __syncthreads();
  
  dXSq = data[threadIdx.x % 32];
  for (int s=16; s>0; s/=2) {
    dXSq += __shfl_down_sync(MASK32, dXSq, s);
  }
  if (threadIdx.x == 0) {
    data[0] = dXSq;
    float tmp = dXSq/(N-1);
    // printf("dXSq:   %f\n", dXSq);
    // printf("var:    %f\n", tmp);
    // printf("stdDev: %f\n", sqrt(tmp));
  }
  
  __syncthreads();
  
  // Distribute the variance to all threads
  float var = data[0]/(N-1);
  float rVar = 1.0/(sqrt(var)+1e-8);
  for (int idx=threadIdx.x; idx<N; idx+=blockDim.x) {
    Y[idx] = (x[idx] + v[idx] + b[idx] - xBar)*rVar*gamma[idx] + beta[idx];
  }
}
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
  
  const dim3 G(1);
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
    
    if (n==0) lbls.push_back("Compute Y");
    t[q++] = omp_get_wtime();
    computeY<<<G, B>>>(pY, px, pv, pb, pgamma, pbeta, vecSize);
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
