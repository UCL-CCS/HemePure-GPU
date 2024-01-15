# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

add_library(hemelb_configuration
	CommandLine.cc
	SimConfig.cc)

set(hip_sources SimConfig.cc)

target_include_directories(hemelb_configuration PUBLIC ${CMAKE_SOURCE_DIR})

if ( USE_HIP )
	set_source_files_properties( SimConfig.cc PROPERTIES LANGUAGE HIP )
	target_sources(hemelb_configuration PRIVATE ${hip_sources} )
endif()
