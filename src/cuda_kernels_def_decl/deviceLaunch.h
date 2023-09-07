#ifndef DEVICE_LAUNCH_H
#define DEVICE_LAUNCH_H

#if defined(HEMELB_USE_HIP)
#include "cuda_kernels_def_decl/hip/kernelLaunch_hip.h"
#elif defined(HEMELB_USE_SYCL)
#include "cuda_kernels_def_decl/sycl/kernelLaunch_sycl.h"
#else
#include "cuda_kernels_def_decl/cuda/kernelLaunch_cuda.h"
#endif

#endif
