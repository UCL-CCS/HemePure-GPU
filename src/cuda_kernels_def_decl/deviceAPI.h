#ifndef DEVICE_API_H
#define DEVICE_API_H

#if defined(HEMELB_USE_HIP)
#include "cuda_kernels_def_decl/hip/deviceAPI_hip.h"
#elif defined(HEMELB_USE_SYCL)
#include "cuda_kernels_def_decl/sycl/deviceAPI_sycl.h"
#else
#include "cuda_kernels_def_decl/cuda/deviceAPI_cuda.h"
#endif


namespace hemelb {
void check_cuda_errors(const char *filename, const int line_number, int myProc);
}

namespace hemelb {

namespace GPU {

const char *deviceGetErrorString();

enum memcpyKind { memcpyHostToDevice, memcpyDeviceToHost };

bool deviceMalloc(void **ptr, size_t MemSz);
bool deviceFree(void *devPtr);

bool deviceMemcpyAsync(void *dst, const void *src, size_t count, memcpyKind kind, Stream_t stream);
bool deviceMemcpy(void *dst, const void *src, size_t count, memcpyKind kind);

#if 0
bool deviceMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset = 0, memcpyKind kind = memcpyHostToDevice);
#endif

bool deviceStreamCreate(Stream_t *streamPtr);
void deviceStreamSynchronize(Stream_t stream);
void deviceStreamDestroy(Stream_t stream);
size_t deviceGetProperties(int myPiD);

int deviceGetCount();
bool deviceAttach(int device);

//char pointerSpace(const void *p);

// IZ July 2024
// TODO: For consistency with Balint's previous efforts - Consider whether these should be transfered in namespace hemelb
bool deviceHostAlloc(void **ptr, size_t MemSz);
bool deviceFreeHost(void *devPtr);


}   // namespace GPU

}   // Namespace hemelb

#endif
