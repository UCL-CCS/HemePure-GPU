#include "cuda_kernels_def_decl/deviceAPI.h"
#include <iostream>

namespace hemelb {
    void check_cuda_errors(const char *filename, const int line_number, int myProc)
    {
    #ifdef DEBUG
        //printf("Debug mode...\n\n");
    //cudaDeviceSynchronize();
        cudaError_t error = cudaGetLastError();
        if(error != cudaSuccess)
        {
            printf("CUDA error at %s:%i: \"%s\" at proc: %i\n", filename, line_number, cudaGetErrorString(error), myProc);
            abort();
            exit(-1);
        }
    #endif
    }
}


const char* deviceGetErrorString()
{
	cudaError_t error = cudaGetLastError();
	return cudaGetErrorString(error);
}

bool deviceMemcpyAsync( void* dst, const void* src, size_t count, memcpyKind kind, Stream_t stream)
{
	cudaMemcpyKind cudaKind = kind == memcpyHostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost ;
	cudaError_t cudaStatus = cudaMemcpyAsync(dst, src, count, cudaKind, stream);
	if ( cudaStatus == cudaSuccess ) {
		return true;
	}
	else {
		return false;
	}
}

bool deviceMemcpy( void* dst, const void* src, size_t count, memcpyKind kind)
{
	cudaMemcpyKind cudaKind = kind == memcpyHostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost ;
	cudaError_t cudaStatus = cudaMemcpy(dst, src, count, cudaKind);
	if ( cudaStatus == cudaSuccess ) {
		return true;
	}
	else {
		return false;
	}
}

bool deviceMalloc(void **ptr, size_t MemSz)
{
	cudaError_t cudaStatus = cudaMalloc(ptr,MemSz);
	if( cudaStatus == cudaSuccess) {
	  	return true;
	}
	else {
		return false;
	}
}

bool deviceHostAlloc(void **ptr, size_t MemSz)
{
  cudaError_t cudaStatus = cudaHostAlloc(ptr,MemSz, cudaHostAllocDefault);
	if( cudaStatus == cudaSuccess) {
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
	cudaMemcpyKind cudaKind = kind == memcpyHostToDevice ? cudaMemcpyHostToDevice : cudaMemcpyDeviceToHost ;
	cudaError_t status = cudaMemcpyToSymbol(symbol,src,count,offset,cudaKind);
	if( status != cudaSuccess ) {
		return false;
	}
	return true;
}

bool deviceStreamCreate(Stream_t* streamPtr)
{

	cudaError_t status = cudaStreamCreate((cudaStream_t *)streamPtr);
	if (status != cudaSuccess ) {
		return false;
	}
	return true;
}

void deviceStreamSynchronize(Stream_t stream)
{
	cudaStreamSynchronize((cudaStream_t)stream);
}

void deviceStreamDestroy(Stream_t stream)
{
	cudaStreamDestroy((cudaStream_t)stream);
}

bool deviceFree(void *devPtr)
{
	cudaError_t ret = cudaFree(devPtr);
	if( ret == cudaSuccess) {
		return true;
	}
	else {
		return false;
	}
}

bool deviceFreeHost(void *devPtr)
{
	cudaError_t ret = cudaFreeHost(devPtr);
	if( ret == cudaSuccess) {
		return true;
	}
	else {
		return false;
	}
}


size_t deviceGetProperties(int myProc)
{
	cudaDeviceProp dev_prop;

	// Just obtain the properties of GPU assigned to task 1
	cudaGetDeviceProperties( &dev_prop, 0);
	hemelb::check_cuda_errors(__FILE__, __LINE__, myProc);

	// Rank 1 only reports:
	if(myProc == 1){
		std::cout << "===============================================" << "\n";
		std::cout << "Device properties: " << std::endl;
		printf("Device name:        %s\n", dev_prop.name);
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
	cudaGetDeviceCount(&dev_count);
	return dev_count;
}

bool deviceAttach(int device)
{
	cudaError_t cudaStatus = cudaSetDevice(device);
	if (cudaStatus != cudaSuccess) {
		return false;
	}
	return true;
}
