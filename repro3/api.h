#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdint>

typedef double distribn_t;
typedef int64_t site_t;
typedef unsigned Direction;
#define local_iolets_MaxSIZE 90

#define GPU_KERNEL __device__
#define GPU_DEVICE_FUNCTION __device__
#define GPU_INLINE_DEVICE_FUNCTION __device__ __forceinline__
namespace hemelb {


	namespace lattices
	{
		class D3Q19 {
			public:
				// The number of discrete velocity vectors
				static const Direction NUMVECTORS = 19;

				// The x, y and z components of each of the discrete velocity vectors
				static const int CX[NUMVECTORS];
				static const int CY[NUMVECTORS];
				static const int CZ[NUMVECTORS];

				// the same in double (in order to prevent int->double conversions), and aligned to 16B
				static const distribn_t CXD[NUMVECTORS] __attribute__((aligned(16)));
				static const distribn_t CYD[NUMVECTORS] __attribute__((aligned(16)));
				static const distribn_t CZD[NUMVECTORS] __attribute__((aligned(16)));


				static const int* discreteVelocityVectors[3];

				static const double EQMWEIGHTS[NUMVECTORS] __attribute__((aligned(16)));

				// The index of the inverse direction of each discrete velocity vector
				static const Direction INVERSEDIRECTIONS[NUMVECTORS];
		};

		const int D3Q19::CX[] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0 };
		const int D3Q19::CY[] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1 };
		const int D3Q19::CZ[] = { 0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1 };

		const distribn_t D3Q19::CXD[] = { 0.0, 1.0, -1.0, 0.0,  0.0, 0.0,  0.0, 1.0, -1.0,  1.0, 
			-1.0, 1.0, -1.0,  1.0, -1.0, 0.0,  0.0,  0.0,  0.0 };
		const distribn_t D3Q19::CYD[] = { 0.0, 0.0,  0.0, 1.0, -1.0, 0.0,  0.0, 1.0, -1.0, -1.0,
			1.0, 0.0,  0.0,  0.0,  0.0, 1.0, -1.0,  1.0, -1.0 };
		const distribn_t D3Q19::CZD[] = { 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, -1.0, 0.0,  0.0,  0.0,
			0.0, 1.0, -1.0, -1.0,  1.0, 1.0, -1.0, -1.0,  1.0 };

		const int* D3Q19::discreteVelocityVectors[] = { CX, CY, CZ };

		const distribn_t D3Q19::EQMWEIGHTS[] = { 1.0 / 3.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
			1.0 / 18.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
			1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
			1.0 / 36.0 };

		const Direction D3Q19::INVERSEDIRECTIONS[] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17 };
		struct D3Q19GPUConstants {

			// The number of discrete velocity vectors
			const Direction NUMVECTORS;
			const int CX[19];
			const int CY[19];
			const int CZ[19];

			const distribn_t EQMWEIGHTS[19];
			const Direction INVERSEDIRECTIONS[19];

			GPU_KERNEL D3Q19GPUConstants() : NUMVECTORS(19),
			CX{0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0},
			CY{0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1},
			CZ{0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1},
			EQMWEIGHTS{1.0 / 3.0,  1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
				1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
				1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0},
			INVERSEDIRECTIONS{0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17} {}
		};
	} // namespace lattices


	namespace GPU {


		// Stream abstraction 
		using Stream_t = ::hipStream_t;
		enum memcpyKind { memcpyHostToDevice, memcpyDeviceToHost };

		// Dispatcher
		template <typename Functor>
			__global__ void
			dispatchWrapper(Functor f) {
				unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
				f(Ind);
			}

		// Dispatch
		template <typename Functor>
			void
			kernelLaunch(Functor &f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {
				dispatchWrapper<Functor><<<NumBlocks, NumThreadsPerBlock, SMem, stream>>>(f);
			}

		namespace hemelb {
			void
				check_cuda_errors(const char *filename, const int line_number, int myProc) {
#ifdef DEBUG
					// printf("Debug mode...\n\n");
					// cudaDeviceSynchronize();
					hipError_t error = hipGetLastError();
					if (error != hipSuccess) {
						printf("CUDA error at %s:%i: \"%s\" at proc: %i\n", filename, line_number, hipGetErrorString(error), myProc);
						abort();
						exit(-1);
					}
#endif
				}
		}   // namespace hemelb

		const char *
			deviceGetErrorString() {
				hipError_t error = hipGetLastError();
				return hipGetErrorString(error);
			}

		bool
			deviceMemcpyAsync(void *dst, const void *src, size_t count, memcpyKind kind, Stream_t stream) {
				hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
				hipError_t hipStatus = hipMemcpyAsync(dst, src, count, hipKind, stream);
				if (hipStatus == hipSuccess) {
					return true;
				} else {
					return false;
				}
			}

		bool
			deviceMemcpy(void *dst, const void *src, size_t count, memcpyKind kind) {
				hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
				hipError_t hipStatus = hipMemcpy(dst, src, count, hipKind);
				if (hipStatus == hipSuccess) {
					return true;
				} else {
					return false;
				}
			}

		bool
			deviceMalloc(void **ptr, size_t MemSz) {
				hipError_t hipStatus = hipMalloc(ptr, MemSz);
				if (hipStatus == hipSuccess) {
					return true;
				} else {
					return false;
				}
			}

		bool
			deviceMemcpyToSymbol(const void *symbol, const void *src, size_t count, size_t offset, memcpyKind kind) {
				hipMemcpyKind hipKind = kind == memcpyHostToDevice ? hipMemcpyHostToDevice : hipMemcpyDeviceToHost;
				hipError_t status = hipMemcpyToSymbol(symbol, src, count, offset, hipKind);
				if (status != hipSuccess) {
					return false;
				}
				return true;
			}

		bool
			deviceStreamCreate(Stream_t *streamPtr) {

				hipError_t status = hipStreamCreate((hipStream_t *) streamPtr);
				if (status != hipSuccess) {
					return false;
				}
				return true;
			}

		void
			deviceStreamSynchronize(Stream_t stream) {
				hipStreamSynchronize((hipStream_t) stream);
			}

		void
			deviceStreamDestroy(Stream_t stream) {
				hipStreamDestroy((hipStream_t) stream);
			}

		bool
			deviceFree(void *devPtr) {
				hipError_t ret = hipFree(devPtr);
				if (ret == hipSuccess) {
					return true;
				} else {
					return false;
				}
			}

		size_t
			deviceGetProperties(int myProc) {
				hipDeviceProp_t dev_prop;

				// Just obtain the properties of GPU assigned to task 1
				hipGetDeviceProperties(&dev_prop, 0);
				hemelb::check_cuda_errors(__FILE__, __LINE__, myProc);

				// Rank 1 only reports:
				if (myProc == 0) {
					std::cout << "==============================================="
						<< "\n";
					std::cout << "Device properties: " << std::endl;
					std::string arch;

					printf("Device name:  AMDGCN GFX%d\n", dev_prop.gcnArch);
					printf("Compute Capability: %d.%d\n\n", dev_prop.major, dev_prop.minor);
					printf("Total Global Mem:    %.1fGB\n", ((double) dev_prop.totalGlobalMem / 1073741824.0));
					std::cout << "Number of Streaming Multiprocessors:  " << dev_prop.multiProcessorCount << std::endl;
					printf("Shared Mem Per SM:   %.0fKB\n", ((double) dev_prop.sharedMemPerBlock / 1024));
					// cout << "Clock Rate:  "<< dev_prop.clockRate<< endl;
					std::cout << "Max Number of Threads per Block:  " << dev_prop.maxThreadsPerBlock << std::endl;
					std::cout << "Max Number of Blocks allowed in x-dir:  " << dev_prop.maxGridSize[0] << std::endl;
					std::cout << "Max Number of Blocks allowed in y-dir:  " << dev_prop.maxGridSize[1] << std::endl;
					std::cout << "Warp Size:  " << dev_prop.warpSize << std::endl;
					std::cout << "==============================================="
						<< "\n\n";
					fflush(stdout);
				}
				return dev_prop.totalGlobalMem;
			}

		int
			deviceGetCount() {
				int dev_count;
				hipGetDeviceCount(&dev_count);
				return dev_count;
			}

		bool
			deviceAttach(int device) {
				hipError_t hipStatus = hipSetDevice(device);
				if (hipStatus != hipSuccess) {
					return false;
				}
				return true;
			}

	};   // Namespace GPU
};   // namespace hemelb


