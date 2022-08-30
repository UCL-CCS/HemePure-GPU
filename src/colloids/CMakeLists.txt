# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.
find_package(MPI REQUIRED)
message(STATUS "MPI Dirs are" ${MPI_INCLUDE_DIRS})

set_source_files_properties(BoundaryConditions.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(BoundaryConditions.cc PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}")
add_library(hemelb_colloidsBCs
	BoundaryConditions.cc)
target_link_libraries(hemelb_colloidsBCs
	hemelb_io MPI::MPI_CXX)

set_source_files_properties(BodyForces.cc PROPERTIES LANGUAGE HIP)
set_source_files_properties(BodyForces.cc PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}")
add_library(hemelb_colloidsForces
	BodyForces.cc)
target_link_libraries(hemelb_colloidsForces
	hemelb_io MPI::MPI_CXX)

set(hemelb_colloids_sources 
	PersistedParticle.cc
)
set(hemelb_colloids_hip_sources
	ParticleMpiDatatypes.cc
	ParticleSet.cc
	Particle.cc
	ColloidController.cc
)

set_source_files_properties(
	${hemelb_colloids_hip_sources}
	PROPERTIES LANGUAGE HIP
)

set_source_files_properties(
  ${hemelb_colloids_hip_sources}
  PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_DIRS}")
	
add_library(hemelb_colloids 
	${hemelb_colloids_sources}
	${hemelb_colloids_hip_sources}
)

target_link_libraries(hemelb_colloids
	hemelb_io
	hemelb_lb
	hemelb_colloidsForces
	hemelb_colloidsBCs
  MPI::MPI_CXX)
