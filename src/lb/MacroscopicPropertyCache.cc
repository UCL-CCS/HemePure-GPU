
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/MacroscopicPropertyCache.h"

namespace hemelb
{
  namespace lb
  {
    MacroscopicPropertyCache::MacroscopicPropertyCache(const SimulationState& simState,
                                                       const geometry::LatticeData& latticeData) :
      densityCache(simState, latticeData.GetLocalFluidSiteCount()),
      velocityCache(simState, latticeData.GetLocalFluidSiteCount()),
      wallShearStressMagnitudeCache(simState, latticeData.GetLocalFluidSiteCount()),
      vonMisesStressCache(simState, latticeData.GetLocalFluidSiteCount()),
      shearRateCache(simState, latticeData.GetLocalFluidSiteCount()),
      stressTensorCache(simState, latticeData.GetLocalFluidSiteCount()),
      tractionCache(simState, latticeData.GetLocalFluidSiteCount()),
      tangentialProjectionTractionCache(simState, latticeData.GetLocalFluidSiteCount()),
      siteCount(latticeData.GetLocalFluidSiteCount()),
      wallMom_Cache(simState, latticeData.GetLocalFluidSiteCount()*latticeData.GetLatticeInfo().GetNumVectors()), // Change this in the future with a function that will return the max of the IOLETs sites... TODO
      wallMom_correction_Cache(simState, latticeData.GetLocalFluidSiteCount()*latticeData.GetLatticeInfo().GetNumVectors())
    {
      ResetRequirements();
    }

    void MacroscopicPropertyCache::ResetRequirements()
    {
      densityCache.UnsetRefreshFlag();
      velocityCache.UnsetRefreshFlag();
      wallMom_Cache.UnsetRefreshFlag();
      wallMom_correction_Cache.UnsetRefreshFlag();
      vonMisesStressCache.UnsetRefreshFlag();
      wallShearStressMagnitudeCache.UnsetRefreshFlag();
      shearRateCache.UnsetRefreshFlag();
      stressTensorCache.UnsetRefreshFlag();
      tractionCache.UnsetRefreshFlag();
      tangentialProjectionTractionCache.UnsetRefreshFlag();
    }

    site_t MacroscopicPropertyCache::GetSiteCount() const
    {
      return siteCount;
    }
  }
}
