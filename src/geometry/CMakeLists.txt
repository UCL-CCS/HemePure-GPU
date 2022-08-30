# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

set(hemelb_geometry_sources
	Block.cc
	SiteData.cc
	SiteDataBare.cc
	VolumeTraverser.cc
	decomposition/BasicDecomposition.cc
	neighbouring/NeighbouringLatticeData.cc
	neighbouring/RequiredSiteInformation.cc
)

set(hemelb_geometry_hip_sources
	BlockTraverser.cc
	BlockTraverserWithVisitedBlockTracker.cc
	GeometryReader.cc
	LatticeData.cc
	SiteTraverser.cc
	decomposition/OptimisedDecomposition.cc
	needs/Needs.cc
	neighbouring/NeighbouringDataManager.cc
)
set_source_files_properties(${hemelb_geometry_hip_sources}
	PROPERTIES LANGUAGE HIP
)
set_source_files_properties(${hemelb_geometry_hip_sources}
  PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}")

add_library(hemelb_geometry 
	${hemelb_geometry_sources}
	${hemelb_geometry_hip_sources}
)
target_link_libraries(hemelb_geometry PUBLIC MPI::MPI_CXX)

configure_file(
	"${PROJECT_SOURCE_DIR}/geometry/decomposition/DecompositionWeights.h.in"
	"${PROJECT_BINARY_DIR}/geometry/decomposition/DecompositionWeights.h")
