# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

set(hemelb_lb_sources
	iolets/BoundaryCommunicator.cc
	iolets/InOutLet.cc
	iolets/InOutLetCosine.cc
	iolets/InOutLetFile.cc
	iolets/InOutLetVelocity.cc
	iolets/InOutLetParabolicVelocity.cc
	iolets/InOutLetWomersleyVelocity.cc
	iolets/InOutLetFileVelocity.cc
	kernels/momentBasis/DHumieresD3Q15MRTBasis.cc
	kernels/momentBasis/DHumieresD3Q19MRTBasis.cc
	kernels/rheologyModels/AbstractRheologyModel.cc
	kernels/rheologyModels/CarreauYasudaRheologyModel.cc
	kernels/rheologyModels/CassonRheologyModel.cc
	kernels/rheologyModels/TruncatedPowerLawRheologyModel.cc
	lattices/LatticeInfo.cc
	lattices/D3Q15.cc
	lattices/D3Q19.cc
	lattices/D3Q27.cc
	lattices/D3Q15i.cc
	SimulationState.cc
)

set( hemelb_lb_hip_sources 
	iolets/BoundaryComms.cc 
	iolets/BoundaryValues.cc 
	IncompressibilityChecker.cc
	MacroscopicPropertyCache.cc
	StabilityTester.cc
)
set_source_files_properties(${hemelb_lb_sources} PROPERTIES LANGUAGE CXX)
set_source_files_properties(${hemelb_lb_hip_sources} PROPERTIES LANGUAGE HIP)
set_source_files_properties(${hemelb_lb_hip_sources} PROPERTIES INCLUDE_DIRECTORIES "${MPI_CXX_COMPILER_INCLUDE_DIRS}")

add_library(hemelb_lb ${hemelb_lb_sources} ${hemelb_lb_hip_sources})
target_link_libraries(hemelb_lb PUBLIC MPI::MPI_CXX)


