#ifndef HIP_KERNEL_LAUNCH_H
#define HIP_KERNEL_LAUNCH_H

#include <hip/hip_runtime.h>

#define GPU_KERNEL __host__ __device__
#define GPU_FUNCTION __host__ __device__
#define GPU_INLINE_FUNCTION __host__ __device__ __forceinline__

#ifndef HIP_DEVICE_API_H
#include "cuda_kernels_def_decl/hip/deviceAPI_hip.h"
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
kernelLaunch(Functor &f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {
  dispatchWrapper<Functor><<<NumBlocks, NumThreadsPerBlock, SMem, stream>>>(f);
}
template <typename Functor>
void
kernelLaunchStrided(Functor &f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {
  dispatchWrapperStrided<Functor><<<NumBlocks, NumThreadsPerBlock, SMem, stream>>>(f);
}

}   // namespace GPU
}   // namespace hemelb

#endif
