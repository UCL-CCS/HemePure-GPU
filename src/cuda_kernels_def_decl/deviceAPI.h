#ifndef DEVICE_API_H
#define DEVICE_API_H


#ifdef HEMELB_USE_HIP
#include <hip/hip_runtime.h>
using Stream_t = hipStream_t;
#else
#include <cuda_runtime.h>
using Stream_t = cudaStream_t;
#endif

namespace hemelb {
    void check_cuda_errors(const char *filename, const int line_number, int myProc);
}

const char* deviceGetErrorString();

enum memcpyKind { memcpyHostToDevice , memcpyDeviceToHost };


bool deviceMalloc(void **ptr, size_t MemSz);
bool deviceFree(void *devPtr);
bool deviceFreeHost(void *devPtr);

bool deviceMemcpyAsync( void* dst, const void* src, size_t count, memcpyKind kind, Stream_t stream = 0);
bool deviceMemcpy( void* dst, const void* src, size_t count, memcpyKind kind);
bool deviceMemcpyToSymbol( const void* symbol, const void* src,
							size_t count, size_t offset = 0,
							memcpyKind kind = memcpyHostToDevice);

bool deviceStreamCreate(Stream_t* streamPtr);
void deviceStreamSynchronize(Stream_t stream);
void deviceStreamDestroy(Stream_t stream);
size_t deviceGetProperties(int myPiD);

int deviceGetCount();
bool deviceAttach(int device);

bool deviceHostAlloc(void **ptr, size_t MemSz);

#endif
