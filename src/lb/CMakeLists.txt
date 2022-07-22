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

add_custom_command(
  OUTPUT lb.h
  OUTPUT lb.hpp
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND
  cp $<$<BOOL:${USE_CUDA}>:lb_cuda.h>
     $<$<BOOL:${USE_HIP_ROCM}>:lb_hip.h>
     $<$<BOOL:${USE_HIP_CUDA}>:lb_hip.h>
     lb.h
  COMMAND
  cp $<$<BOOL:${USE_CUDA}>:lb_cuda.hpp>
     $<$<BOOL:${USE_HIP_ROCM}>:lb_hip.hpp>
     $<$<BOOL:${USE_HIP_CUDA}>:lb_hip.hpp>
     lb.hpp)
add_custom_target(lb_header ALL DEPENDS lb.h lb.hpp)
add_dependencies(hemelb_lb lb_header)
