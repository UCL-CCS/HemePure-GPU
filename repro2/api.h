#pragma once
#include <iostream>
#include <cstdint>
#include <sycl/sycl.hpp>

#define GPU_KERNEL 
#define GPU_DEVICE_FUNCTION 
#define GPU_INLINE_DEVICE_FUNCTION __forceinline__


namespace hemelb {

void
check_cuda_errors(const char *filename, const int line_number, int myProc) {
	// Figure out what to do here
}

namespace {

	class StreamManager {
	public:
		using Stream_t = unsigned int;
		using StreamInternal_t = sycl::queue;
		static constexpr Stream_t default_stream = 0;

		StreamManager(sycl::device& dev) : theDevice( dev ) {
			theContextPtr.reset( new sycl::context( theDevice ) );
			newStream(); // Add default stream
		}		  


		Stream_t newStream() {
		  Stream_t ret_val = 0;

		  if ( freeList.empty() ) {
			// No previous slot to reuse -- push on to the end
			theStreams.push_back( std::make_unique<sycl::queue>( *theContextPtr , theDevice, sycl::property::queue::in_order() ) );
		 	ret_val = theStreams.size()-1;	
		  }
		  else {
			// reuse from the free list

			// Take the last index off the free list
			ret_val = freeList.back();
			freeList.pop_back();

			// reuse the nullptr there
			theStreams[ret_val].reset( new sycl::queue( *theContextPtr , theDevice, sycl::property::queue::in_order() ) );
		  }
		  return ret_val;
		}

		bool eraseStream(Stream_t stream) {
			 if( stream < theStreams.size() ) {
			 	theStreams[stream].reset(nullptr);
			 	freeList.push_back(stream);
				return true;
			 }
			 else { 
				std::cerr << "Attempt to delete stream with index outside allowed range\n";
				return false; 
			}
		}

		StreamInternal_t&  getStream(Stream_t stream) {
			if( stream >= theStreams.size() ) {
				std::cerr << "Attempt to access stream outside of allowed range\n";
				std::abort();
			}
			if( theStreams[stream] == nullptr ) {
				std::cerr << "Attempt to access invalid( possibly deleted ) stream: " << stream << "\n";
				std::abort();
			}
			return (*theStreams[stream]);
		}
			
		StreamInternal_t& getDefaultStream() {
			return (*theStreams[ default_stream ]);
		}

	private: 
		sycl::device& theDevice;
		std::unique_ptr<sycl::context> theContextPtr{nullptr};
		std::vector<std::unique_ptr<sycl::queue>> theStreams;
		std::vector<unsigned int> freeList={};
	};


	static sycl::device _theDevice;
	static bool _inited = false;
	static std::unique_ptr<StreamManager> _streamManager;
} // End anonymous namespace


namespace GPU {  // hemelb::GPU

using Stream_t = StreamManager::Stream_t;
enum memcpyKind { memcpyHostToDevice, memcpyDeviceToHost };

	
// Dispatch
template <typename Functor>
void
kernelLaunch(Functor f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {
	auto& queue = _streamManager->getStream(stream);

	// Dumb flat dispatch for now
	size_t MaxRange = NumBlocks*NumThreadsPerBlock;

	queue.submit([&]( sycl::handler& cgh ) {
		cgh.parallel_for(sycl::nd_range<1>({MaxRange}, {NumThreadsPerBlock}),
	  	 [=]( sycl::nd_item<1> idx ) {
						// Turn sycl::id into an index our kernels can undestand
			  unsigned long long Ind = static_cast<unsigned long long>( idx.get_global_id(0) );
	    	  f(Ind);
		});

	});
	queue.wait(); // Dunno if I want this here, or just leave it in the explicit synchronize method
}

int
deviceGetCount() {
   return sycl::device::get_devices(sycl::info::device_type::gpu).size();
}

bool deviceAttach(unsigned int deviceID)
{
  if( !_inited ) {
	auto dlist = sycl::device::get_devices(sycl::info::device_type::gpu);
	if ( dlist.size() == 0 ) {
		fprintf(stderr, "No GPU Devices available\n");
		return false;
	}
	if( deviceID > dlist.size() ) {
		fprintf(stderr, "DeviceID > Number of available devices\n");
		return false;
	}
	
	_theDevice = dlist[deviceID];
	_streamManager = std::make_unique<StreamManager>(_theDevice);
	_inited = true;
	return true;
   }
  else return false;

}	


const char *
deviceGetErrorString() {
  return "not implemented";
}


bool
deviceMemcpyAsync(void *dst, const void *src, size_t count, memcpyKind kind, Stream_t stream) {
  auto q = _streamManager->getStream(stream);
  // for now ignore memcpyKind... It really should depend on the pointers
  q.memcpy(dst, src, count);
  return true;
}

bool
deviceMemcpy(void *dst, const void *src, size_t count, memcpyKind kind) {
  auto q = _streamManager->getDefaultStream();
  // for now ignore memcpyKind... It really should depend on the pointers
  q.memcpy(dst, src, count);
  q.wait();
  return true;
}

bool
deviceMalloc(void **ptr, size_t MemSz) {
  auto q = _streamManager->getDefaultStream();
  *ptr = sycl::malloc_device( MemSz, q );
   if( ! ptr ) return false;
	else return true;
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
	*streamPtr = _streamManager->newStream();
	return true;
}

void
deviceStreamSynchronize(Stream_t stream) {
  auto& s = _streamManager->getStream(stream);
  s.wait();
}

void
deviceStreamDestroy(Stream_t stream) {
	_streamManager->eraseStream(stream);
}

bool
deviceFree(void *devPtr) {
  auto q = _streamManager->getDefaultStream();
  sycl::free( devPtr, q );
  return true;
}

		size_t
			deviceGetProperties(int myProc) {

				// Rank 1 only reports:
				if (myProc == 0) {
					std::cout << "==============================================="
						<< "\n";
					std::cout << "Device properties: " << std::endl;
				   	std::cout << "   Name: " << _theDevice.get_info< sycl::info::device::vendor>()
							  << " " << _theDevice.get_info< sycl::info::device::name>() << "\n";
					std::cout << "   Total Global Mem:    " << _theDevice.get_info< sycl::info::device::global_mem_size >()/(1024*1024*1024) << " GiB\n";
					std::cout << "   Local Mem Size:  " << _theDevice.get_info< sycl::info::device::local_mem_size >()/1024 << " KiB\n";
					std::cout << "   Cache Size: " <<  _theDevice.get_info< sycl::info::device::global_mem_cache_size >()/1024 << " KiB\n";
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
				return _theDevice.get_info< sycl::info::device::global_mem_size>();
			}
};   // Namespace GPU


typedef double distribn_t;
typedef int64_t site_t;
typedef unsigned Direction;
#define local_iolets_MaxSIZE 90

	struct Iolets{
		int n_local_iolets;						// 	Number of local Rank Iolets - NOTE: Some Iolet IDs may repeat, depending on the fluid ID numbering - see the value of unique iolets, (for example n_unique_LocalInlets_mInlet_Edge)
		site_t Iolets_ID_range[local_iolets_MaxSIZE]; 	//	Iolet ID and fluid sites range: [min_Fluid_Index, max_Fluid_Index], i.e 3 site_t values per iolet
	};


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

};   // namespace hemelb


