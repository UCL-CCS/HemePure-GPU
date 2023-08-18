#ifndef DEVICE_LAUNCH_H
#define DEVICE_LAUNCH_H

#ifdef HEMELB_USE_HIP
#include "cuda_kernels_def_decl/hip/kernelLaunch_hip.h"
#else 
#include "cuda_kernels_def_decl/cuda/kernelLaunch_cuda.h"
#endif

#endif
