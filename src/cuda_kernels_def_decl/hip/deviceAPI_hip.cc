#include "cuda_kernels_def_decl/deviceAPI.h"
#include <iostream>

namespace hemelb {
    void check_cuda_errors(const char *filename, const int line_number, int myProc)
    {
    #ifdef DEBUG
        //printf("Debug mode...\n\n");
    //cudaDeviceSynchronize();
        hipError_t error = hipGetLastError();
        if(error != hipSuccess)
        {
            printf("CUDA error at %s:%i: \"%s\" at proc: %i\n", filename, line_number, hipGetErrorString(error), myProc);
            abort();
            exit(-1);
        }
    #endif
    }
}


const char* deviceGetErrorString()
{
	hipError_t error = hipGetLastError();
	return hipGetErrorString(error);
}

bool deviceMemcpyAsync( void* dst, const void* src, size_t count, memcpyKind kind, Stream_t stream)
{
	hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost ;
	hipError_t hipStatus = hipMemcpyAsync(dst, src, count, hipKind, stream);
	if ( hipStatus == hipSuccess ) {
		return true;
	}
	else {
		return false;
	}
}

bool deviceMemcpy( void* dst, const void* src, size_t count, memcpyKind kind)
{
	hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost ;
	hipError_t hipStatus = hipMemcpy(dst, src, count, hipKind);
	if ( hipStatus == hipSuccess ) {
		return true;
	}
	else {
		return false;
	}
}

bool deviceMalloc(void **ptr, size_t MemSz)
{
	hipError_t hipStatus = hipMalloc(ptr,MemSz);
	if( hipStatus == hipSuccess) {
	  	return true;
	}
	else {
		return false;
	}
}

bool deviceHostAlloc(void **ptr, size_t MemSz)
{
  hipError_t hipStatus = hipHostMalloc(ptr,MemSz, hipHostMallocDefault);
	if( hipStatus == hipSuccess) {
	  	return true;
	}
	else {
		return false;
	}
}

bool deviceMemcpyToSymbol( const void* symbol, const void* src,
							size_t count, size_t offset,
							memcpyKind kind)
{
	hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost ;
	hipError_t status = hipMemcpyToSymbol(symbol,src,count,offset,hipKind);
	if( status != hipSuccess ) {
		return false;
	}
	return true;
}

bool deviceStreamCreate(Stream_t* streamPtr)
{

	hipError_t status = hipStreamCreate((hipStream_t *)streamPtr);
	if (status != hipSuccess ) {
		return false;
	}
	return true;
}

void deviceStreamSynchronize(Stream_t stream)
{
	hipStreamSynchronize((hipStream_t)stream);
}

void deviceStreamDestroy(Stream_t stream)
{
	hipStreamDestroy((hipStream_t)stream);
}

bool deviceFree(void *devPtr)
{
	hipError_t ret = hipFree(devPtr);
	if( ret == hipSuccess) {
		return true;
	}
	else {
		return false;
	}
}

bool deviceFreeHost(void *devPtr)
{
	hipError_t ret = hipHostFree(devPtr);
	if( ret == hipSuccess) {
		return true;
	}
	else {
		return false;
	}
}

size_t deviceGetProperties(int myProc)
{
	hipDeviceProp_t dev_prop;

	// Just obtain the properties of GPU assigned to task 1
	hipGetDeviceProperties( &dev_prop, 0);
	hemelb::check_cuda_errors(__FILE__, __LINE__, myProc);

	// Rank 1 only reports:
	if( myProc==1){
		std::cout << "===============================================" << "\n";
		std::cout << "Device properties: " << std::endl;
		std::string arch;

		printf("Device name:  AMDGCN GFX%d\n", dev_prop.gcnArchName); // Changed dev_prop.gcnArch to dev_prop.gcnArchName
		printf("Compute Capability: %d.%d\n\n", dev_prop.major, dev_prop.minor);
		printf("Total Global Mem:    %.1fGB\n", ((double)dev_prop.totalGlobalMem/1073741824.0));
		std::cout << "Number of Streaming Multiprocessors:  "<< dev_prop.multiProcessorCount<< std::endl;
		printf("Shared Mem Per SM:   %.0fKB\n", ((double)dev_prop.sharedMemPerBlock/1024));
		//cout << "Clock Rate:  "<< dev_prop.clockRate<< endl;
		std::cout << "Max Number of Threads per Block:  "<< dev_prop.maxThreadsPerBlock << std::endl;
		std::cout << "Max Number of Blocks allowed in x-dir:  "<< dev_prop.maxGridSize[0]<< std::endl;
		std::cout << "Max Number of Blocks allowed in y-dir:  "<< dev_prop.maxGridSize[1]<< std::endl;
		std::cout << "Warp Size:  "<< dev_prop.warpSize<< std::endl;
		std::cout << "===============================================" << "\n\n";
		fflush(stdout);
	}
	return dev_prop.totalGlobalMem;

}

int deviceGetCount()
{
	int dev_count;
	hipGetDeviceCount(&dev_count);
	return dev_count;
}

bool deviceAttach(int device)
{
	hipError_t hipStatus = hipSetDevice(device);
	if (hipStatus != hipSuccess) {
		return false;
	}

	return true;
}
