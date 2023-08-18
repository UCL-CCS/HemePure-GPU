#ifndef DEVICE_API_CUDA_H
#define DEVICE_API_CUDA_H

#include <cuda_runtime.h>

namespace hemelb {
namespace GPU {

using Stream_t = cudaStream_t;

}   // namespace GPU
}   // namespace hemelb

#endif
