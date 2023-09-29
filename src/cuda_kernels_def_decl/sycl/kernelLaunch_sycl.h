#pragma once
#include <iostream>
#include <cstdint>
#include <sycl/sycl.hpp>
#include "cuda_kernels_def_decl/sycl/deviceAPI_sycl.h"

#define GPU_KERNEL 
#define GPU_DEVICE_FUNCTION 
#define GPU_INLINE_DEVICE_FUNCTION inline
#define GPU_DUMMY_SYNC
namespace hemelb {
namespace GPU {  // hemelb::GPU

// Dispatch
template <typename Functor>
void
kernelLaunch(Functor f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {
//	std::cout << "Launching Kernel: " << typeid(f).name() << "\n";
	auto& queue = Impl::getStreamManager().getStream(stream);

	// Dumb flat dispatch for now
	size_t MaxRange = NumBlocks*NumThreadsPerBlock;

	queue.submit([&]( sycl::handler& cgh ) {
		cgh.parallel_for(sycl::nd_range<1>({MaxRange}, {NumThreadsPerBlock}),
	  	 [=]( sycl::nd_item<1> idx ) {
			  unsigned long long Ind = static_cast<unsigned long long>( idx.get_global_id(0) );
	    	  f(Ind);
		});

	});
}

template <typename Functor>
void
kernelLaunchStrided(Functor f, size_t NumBlocks, size_t NumThreadsPerBlock, size_t SMem, Stream_t stream = (Stream_t) 0) {

//	std::cout << "Launching Strided Kernel: " << typeid(f).name() << "\n";
	auto& queue = Impl::getStreamManager().getStream(stream);

	// Dumb flat dispatch for now
	size_t MaxRange = NumBlocks*NumThreadsPerBlock;

	queue.submit([&]( sycl::handler& cgh ) {
		cgh.parallel_for(sycl::nd_range<1>({MaxRange}, {NumThreadsPerBlock}),
	  	 [=]( sycl::nd_item<1> idx ) {
						// Turn sycl::id into an index our kernels can undestand
			  unsigned long long Ind = static_cast<unsigned long long>( idx.get_global_id(0) );
	    	  f(Ind,MaxRange);
		});

	});
}
}// GPU
} // Hemelb
