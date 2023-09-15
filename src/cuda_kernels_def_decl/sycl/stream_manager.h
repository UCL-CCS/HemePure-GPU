#pragma once
#include <iostream>
#include <cstdint>
#include <sycl/sycl.hpp>

namespace hemelb {

namespace GPU {

namespace Impl {

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
	
	sycl::device& getTheDevice();
	StreamManager& getStreamManager();

	int getGPUCount();
	bool deviceAttach(int deviceID);
}; // Impl
}; // GPU
}; // Hemelb

