#include <iostream>
#include <cstdint>
#include "cuda_kernels_def_decl/sycl/deviceAPI_sycl.h"

namespace hemelb {

void
check_cuda_errors(const char *filename, const int line_number, int myProc) {
	// Figure out what to do here
}

namespace GPU {  // hemelb::GPU

enum memcpyKind { memcpyHostToDevice, memcpyDeviceToHost };

int
deviceGetCount() {
	return Impl::getGPUCount();
}

bool deviceAttach(int deviceID)
{
	return Impl::deviceAttach(deviceID);
}	

const char *
deviceGetErrorString() {
  return "not implemented";
}


bool
deviceMemcpyAsync(void *dst, const void *src, size_t count, memcpyKind kind, Stream_t stream) {
  auto& q = Impl::getStreamManager().getStream(stream);
  q.memcpy(dst, src, count);
  return true;
}

bool
deviceMemcpy(void *dst, const void *src, size_t count, memcpyKind kind) {
  auto& q = Impl::getStreamManager().getDefaultStream();

  // for now ignore memcpyKind... It really should depend on the pointers
  q.memcpy(dst, src, count);
  q.wait();
  return true;
}

bool
deviceMalloc(void **ptr, size_t MemSz) {
  auto& q = Impl::getStreamManager().getDefaultStream();
  *ptr = sycl::malloc_device( MemSz, q );
   if( ! ptr ) return false;
	else return true;
}


// two-arg overload that creates a default queue under the hood
bool 
deviceHostAlloc(void **ptr, size_t MemSz) {
	static sycl::queue q{ sycl::default_selector{} };
      	*ptr = sycl::malloc_host(MemSz, q);
      	return (*ptr != nullptr);
}

bool 
deviceFreeHost(void *devPtr) {
  static sycl::queue q{ sycl::default_selector{} };
  try {
    sycl::free(devPtr, q);
    return true;
  } catch (const sycl::exception &e) {
    std::cerr << "SYCL free failed: " << e.what() << "\n";
    return false;
  }
}

#if 0
		bool
			deviceMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, memcpyKind kind) {
				hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
				hipError_t status = hipMemcpyToSymbol(symbol, src, count, offset, hipKind);
				if (status != hipSuccess) {
					return false;
				}
				return true;
			}
#endif
bool
deviceStreamCreate(Stream_t *streamPtr) {
	*streamPtr = Impl::getStreamManager().newStream();
	return true;
}

void
deviceStreamSynchronize(Stream_t stream) {
  auto& q = Impl::getStreamManager().getStream(stream);
  q.wait();
}

void
deviceStreamDestroy(Stream_t stream) {
	Impl::getStreamManager().eraseStream(stream);	
}

bool
deviceFree(void *devPtr) {
  auto& q = Impl::getStreamManager().getDefaultStream();
  sycl::free( devPtr, q );
  return true;
}

		size_t
			deviceGetProperties(int myProc) {
				auto dev = Impl::getTheDevice();
	
				// Rank 1 only reports:
				if (myProc == 0) {
					std::cout << "==============================================="
						<< "\n";
					std::cout << "Device properties: " << std::endl;
				   	std::cout << "   Name: " << dev.get_info< sycl::info::device::vendor>()
							  << " " << dev.get_info< sycl::info::device::name>() << "\n";
					std::cout << "   Total Global Mem:    " << dev.get_info< sycl::info::device::global_mem_size >()/(1024*1024*1024) << " GiB\n";
					std::cout << "   Local Mem Size:  " << dev.get_info< sycl::info::device::local_mem_size >()/1024 << " KiB\n";
					std::cout << "   Cache Size: " <<  dev.get_info< sycl::info::device::global_mem_cache_size >()/1024 << " KiB\n";
#if 		0
					std::cout << "   Max Number of Threads per Block:  " << dev_prop.maxThreadsPerBlock << std::endl;
					std::cout << "   Max Number of Blocks allowed in x-dir:  " << dev_prop.maxGridSize[0] << std::endl;
					std::cout << "   Max Number of Blocks allowed in y-dir:  " << dev_prop.maxGridSize[1] << std::endl;
					std::cout << "   Warp Size:  " << _theDevice.get_info< sycl::device::info::
					std::cout << "==============================================="
						<< "\n\n";
					fflush(stdout);
#endif 
				}
				return dev.get_info< sycl::info::device::global_mem_size>();
			}
};   // Namespace GPU

}; // hemelb
