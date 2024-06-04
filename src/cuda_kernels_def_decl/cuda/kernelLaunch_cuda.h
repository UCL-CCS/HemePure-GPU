#ifndef KERNEL_LAUNCH_CUDA_H
#define KERNEL_LAUNCH_CUDA_H

 
#define GPU_KERNEL __host__ __device__
#define GPU_DEVICE_FUNCTION __device__
#define GPU_INLINE_DEVICE_FUNCTION __host__ __device__ __forceinline__

#define GPU_DUMMY_SYNC 


#ifndef DEVICE_API_CUDA_H
#include "cuda_kernels_def_decl/cuda/deviceAPI_cuda.h"
#endif

namespace hemelb {
namespace GPU {

template <typename Functor>
__global__ void

dispatchWrapper(Functor f) {
  unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
  f(Ind);
}

template <typename Functor>
__global__ void
dispatchWrapperStrided(Functor f) {
  unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long Stride = blockDim.x * gridDim.x;
  f(Ind, Stride);
}
// Dispatch
template <typename Functor>
void
kernelLaunch(Functor f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream=0) {


  dispatchWrapper<Functor><<<NumBlocks, NumThreadsPerBlock, SMem, stream>>>(f);
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess) {
    printf("CUDA error: \"%s\" \n", cudaGetErrorString(error));
    fflush(stdout);
    abort();
    exit(-1);
  }
}

template <typename Functor>
void
kernelLaunchStrided(Functor &f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {
  dispatchWrapperStrided<Functor><<<NumBlocks, NumThreadsPerBlock, SMem, stream>>>(f);
}

}   // namespace GPU
}   // namespace hemelb

#endif
