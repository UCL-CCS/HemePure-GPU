# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.
set(hemelb_net_sources
	MpiDataType.cc
	MpiEnvironment.cc
	MpiError.cc
	MpiCommunicator.cc
	MpiGroup.cc
	MpiFile.cc
	IteratedAction.cc
	IOCommunicator.cc
  ProcComms.cc
	phased/StepManager.cc
)

set(hemelb_net_hip_sources
	BaseNet.cc
	mixins/pointpoint/CoalescePointPoint.cc
	mixins/pointpoint/SeparatedPointPoint.cc
	mixins/pointpoint/ImmediatePointPoint.cc
	mixins/gathers/SeparatedGathers.cc
	mixins/gathers/ViaPointPointGathers.cc
	mixins/alltoall/SeparatedAllToAll.cc
	mixins/alltoall/ViaPointPointAllToAll.cc
	mixins/StoringNet.cc 
)

set_source_files_properties(${hemelb_net_hip_sources} 
  PROPERTIES LANGUAGE HIP)

add_library(hemelb_net
 ${hemelb_net_sources}
 ${hemelb_net_hip_sources}
)

configure_file(
	"${PROJECT_SOURCE_DIR}/net/BuildInfo.h.in"
	"${PROJECT_BINARY_DIR}/net/BuildInfo.h")

configure_file(
	"${PROJECT_SOURCE_DIR}/net/MpiConstness.h.in"
	"${PROJECT_BINARY_DIR}/net/MpiConstness.h")
