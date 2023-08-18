
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_LATTICES_D3Q19_CONST_H
#define HEMELB_LB_LATTICES_D3Q19_CONST_H
#include <units.h>
#include "cuda_kernels_def_decl/deviceLaunch.h"

namespace hemelb {
namespace lb {
namespace lattices {
struct D3Q19GPUConstants {

  // The number of discrete velocity vectors
  const Direction NUMVECTORS;
  const int CX[19];
  const int CY[19];
  const int CZ[19];

  const distribn_t EQMWEIGHTS[19];
  const Direction INVERSEDIRECTIONS[19];
 
  GPU_KERNEL D3Q19GPUConstants() : NUMVECTORS(19),
	CX{0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1, 0, 0, 0, 0},
	CY{0, 0, 0, 1, -1, 0, 0, 1, -1, -1, 1, 0, 0, 0, 0, 1, -1, 1, -1},
	CZ{0, 0, 0, 0, 0, 1, -1, 0, 0, 0, 0, 1, -1, -1, 1, 1, -1, -1, 1},
	EQMWEIGHTS{1.0 / 3.0,  1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0, 1.0 / 18.0,
               1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0,
                                       1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0, 1.0 / 36.0},
	INVERSEDIRECTIONS{0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15, 18, 17} {}
};
}   // namespace lattices
}   // namespace lb
}   // namespace hemelb

#endif
