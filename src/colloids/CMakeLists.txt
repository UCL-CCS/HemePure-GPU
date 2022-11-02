# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

add_library(hemelb_colloidsBCs
	BoundaryConditions.cc)
target_link_libraries(hemelb_colloidsBCs
	hemelb_io)
set_source_files_properties(BoundaryConditions.cc PROPERTIES LANGUAGE HIP)

add_library(hemelb_colloidsForces
	BodyForces.cc)
set_source_files_properties(BodyForces.cc PROPERTIES LANGUAGE HIP)
target_link_libraries(hemelb_colloidsForces
	hemelb_io)

add_library(hemelb_colloids
	ParticleMpiDatatypes.cc
	ParticleSet.cc
	Particle.cc
	PersistedParticle.cc
	ColloidController.cc)
target_link_libraries(hemelb_colloids
	hemelb_io
	hemelb_lb
	hemelb_colloidsForces
	hemelb_colloidsBCs)

if( USE_HIP )
set_source_files_properties(ParticleMpiDatatypes.cc
	ParticleSet.cc
	Particle.cc
	ColloidController.cc  PROPERTIES LANGUAGE HIP)
endif()

