# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

add_library(hemelb_net
  MpiDataType.cc
  MpiEnvironment.cc
  MpiError.cc
  MpiCommunicator.cc
  MpiGroup.cc
  MpiFile.cc
  IteratedAction.cc
  $<$<BOOL:${USE_CUDA}>:BaseNet.cc>
  IOCommunicator.cc
  mixins/pointpoint/CoalescePointPoint.cc
  mixins/pointpoint/SeparatedPointPoint.cc
  mixins/pointpoint/ImmediatePointPoint.cc
  mixins/gathers/SeparatedGathers.cc
  mixins/gathers/ViaPointPointGathers.cc
  mixins/alltoall/SeparatedAllToAll.cc
  mixins/alltoall/ViaPointPointAllToAll.cc
  mixins/StoringNet.cc ProcComms.cc
  phased/StepManager.cc)

target_include_directories(hemelb_net PUBLIC ${CMAKE_SOURCE_DIR})

if(USE_HIP) 
  set_source_files_properties(mixins/pointpoint/CoalescePointPoint.cc 
		mixins/pointpoint/SeparatedPointPoint.cc 
		mixins/pointpoint/ImmediatePointPoint.cc
	  mixins/gathers/SeparatedGathers.cc
		mixins/gathers/ViaPointPointGathers.cc
		mixins/alltoall/SeparatedAllToAll.cc
	  mixins/alltoall/ViaPointPointAllToAll.cc
		mixins/StoringNet.cc
		PROPERTIES LANGUAGE HIP)
elseif(USE_SYCL)
  set_source_files_properties(mixins/pointpoint/CoalescePointPoint.cc 
		mixins/pointpoint/SeparatedPointPoint.cc 
		mixins/pointpoint/ImmediatePointPoint.cc
 	    mixins/gathers/SeparatedGathers.cc
		mixins/gathers/ViaPointPointGathers.cc
		mixins/alltoall/SeparatedAllToAll.cc
	    mixins/alltoall/ViaPointPointAllToAll.cc
		mixins/StoringNet.cc
		PROPERTIES COMPILE_FLAGS ${SYCL_CPPFLAGS})
endif()

if(USE_HIP_ROCM OR USE_HIP_CUDA)
  set(hip_sources BaseNet.cc)
  set_source_files_properties(${hip_sources} PROPERTIES LANGUAGE HIP)
  target_sources(hemelb_net PRIVATE ${hip_sources})
elseif( USE_SYCL )
  set(sycl_sources BaseNet.cc)
  set_source_files_properties(${sycl_sources} PROPERTIES COMPILE_FLAGS ${SYCL_CPPFLAGS})
  target_sources(hemelb_net PRIVATE ${sycl_sources})
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/net/BuildInfo.h.in"
  "${PROJECT_BINARY_DIR}/net/BuildInfo.h")

configure_file(
  "${PROJECT_SOURCE_DIR}/net/MpiConstness.h.in"
  "${PROJECT_BINARY_DIR}/net/MpiConstness.h")
