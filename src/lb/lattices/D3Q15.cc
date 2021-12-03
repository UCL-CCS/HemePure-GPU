
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/lattices/D3Q15.h"

namespace hemelb
{
  namespace lb
  {
    namespace lattices
    {
      template<>
      lb::lattices::LatticeInfo* lb::lattices::Lattice<D3Q15>::singletonInfo = NULL;
           
      const int D3Q15::CX[] = { 0, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1, 1, -1 };
      const int D3Q15::CY[] = { 0, 0, 0, 1, -1, 0, 0, 1, -1, 1, -1, -1, 1, -1, 1 };
      const int D3Q15::CZ[] = { 0, 0, 0, 0, 0, 1, -1, 1, -1, -1, 1, 1, -1, -1, 1 };
      const int* D3Q15::discreteVelocityVectors[] = { CX, CY, CZ };

      const distribn_t D3Q15::CXD[] = { 0.0, 1.0, -1.0, 0.0,  0.0, 0.0,  0.0, 1.0, -1.0,  1.0, -1.0,  1.0, -1.0,  1.0, -1.0};
      const distribn_t D3Q15::CYD[] = { 0.0, 0.0,  0.0, 1.0, -1.0, 0.0,  0.0, 1.0, -1.0,  1.0, -1.0, -1.0,  1.0, -1.0,  1.0};
      const distribn_t D3Q15::CZD[] = { 0.0, 0.0,  0.0, 0.0,  0.0, 1.0, -1.0, 1.0, -1.0, -1.0,  1.0,  1.0, -1.0, -1.0,  1.0};            
      
      const distribn_t D3Q15::EQMWEIGHTS[] = { 2.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0 / 9.0, 1.0
          / 9.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0, 1.0 / 72.0 };

      const Direction D3Q15::INVERSEDIRECTIONS[] = { 0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13 };
    }
  }
}
