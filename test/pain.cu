#include <cassert>
#include <stdlib.h>
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <mpi.h>

template <typename Real>
__global__ void getall(Real* cudadata, Real* d_A, Real* d_W) {

        
    // unsigned long tid = blockIdx.x  threadIdx.x;
    for(int i = 0; i<3; ++i)
    {
        cudadata[blockIdx.x*3 + threadIdx.x] = d_A[blockIdx.x*3 + i] * d_A[threadIdx.x*3 + i] * d_W[i*3];
    }
    __syncthreads();
  }