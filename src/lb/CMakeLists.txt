# This file is part of HemeLB and is Copyright (C)
# the HemeLB team and/or their institutions, as detailed in the
# file AUTHORS. This software is provided under the terms of the
# license in the file LICENSE.

add_library(hemelb_lb
	iolets/BoundaryCommunicator.cc
	iolets/BoundaryComms.cc
	iolets/BoundaryValues.cc
	iolets/InOutLet.cc
	iolets/InOutLetCosine.cc
	iolets/InOutLetFile.cc
	iolets/InOutLetVelocity.cc
	iolets/InOutLetParabolicVelocity.cc
	iolets/InOutLetWomersleyVelocity.cc
	iolets/InOutLetFileVelocity.cc
	IncompressibilityChecker.cc
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
	MacroscopicPropertyCache.cc
	SimulationState.cc
	StabilityTester.cc)
target_include_directories(hemelb_lb PUBLIC ${CMAKE_SOURCE_DIR})

if( USE_CUDA )
  set_source_files_properties(iolets/BoundaryComms.cc
      iolets/BoundaryValues.cc
      IncompressibilityChecker.cc
      MacroscopicPropertyCache.cc
      StabilityTester.cc PROPERTIES LANGUAGE CUDA)
elseif( USE_HIP )
   set_source_files_properties(iolets/BoundaryComms.cc
      iolets/BoundaryValues.cc
      IncompressibilityChecker.cc
      MacroscopicPropertyCache.cc
      StabilityTester.cc PROPERTIES LANGUAGE HIP)
elseif( USE_SYCL )
	set_source_files_properties(iolets/BoundaryComms.cc 
			iolets/BoundaryValues.cc 
			IncompressibilityChecker.cc
			MacroscopicPropertyCache.cc  
			StabilityTester.cc PROPERTIES COMPILE_FLAGS ${SYCL_CPPFLAGS})
endif()

