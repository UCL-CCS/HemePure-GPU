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

if(USE_HIP_ROCM OR USE_HIP_CUDA)
  set(hip_sources BaseNet_hip.cc)
  set_source_files_properties(${hip_sources} PROPERTIES HIP_SOURCE_PROPERTY_FORMAT 1)
  hip_add_library(net_hip_objs ${hip_sources})
  target_link_libraries(net_hip_objs PRIVATE $<$<BOOL:${USE_HIP_ROCM}>:hip::device>)
  target_link_libraries(hemelb_net net_hip_objs)
endif()

configure_file(
  "${PROJECT_SOURCE_DIR}/net/BuildInfo.h.in"
  "${PROJECT_BINARY_DIR}/net/BuildInfo.h")

configure_file(
  "${PROJECT_SOURCE_DIR}/net/MpiConstness.h.in"
  "${PROJECT_BINARY_DIR}/net/MpiConstness.h")
