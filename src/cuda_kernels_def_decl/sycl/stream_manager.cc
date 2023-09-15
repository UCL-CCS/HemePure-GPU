#include <iostream>
#include <cstdint>
#include <sycl/sycl.hpp>
#include "cuda_kernels_def_decl/sycl/stream_manager.h"

namespace hemelb {

namespace GPU {

namespace Impl {

	static sycl::device _theDevice;
	static bool _inited = false;
	static std::unique_ptr<StreamManager> _streamManager=nullptr;

	int getGPUCount() {
   		return sycl::device::get_devices(sycl::info::device_type::gpu).size();
	}


	bool deviceAttach(int deviceID)
	{
	  if( deviceID < 0 ) {
	   std::cerr << "ERROR: -ve device ID requested\n";
	   abort();
      }

   	  if( ! _inited ) {
		auto dlist = sycl::device::get_devices(sycl::info::device_type::gpu);
		if ( dlist.size() == 0 ) {
 		  std::cerr << "No GPU Devices available\n";
		  return false;
		}
		if( deviceID > dlist.size() ) {
			std::cerr <<  "DeviceID ("<< deviceID << ") outside the range of available GPU devices (0-" << dlist.size()-1<<")\n";
			return false;
	    }
	
	    _theDevice = dlist[deviceID] ;
	    _streamManager = std::make_unique<StreamManager>(_theDevice);
	    _inited = true;
	   }
	   return _inited;

     }// function

	 StreamManager& getStreamManager() 
	 {
		if( ! _streamManager ) { 
			std::cerr << "Stream Manager Is not initialized\n";
			abort();
		}
		return *_streamManager;
	 }

	 sycl::device& getTheDevice()
	 {
		return _theDevice;
	 }
} // Impl
} // GPU
} // HemeLb
