
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_STREAMERS_LADDIOLETDELEGATE_H
#define HEMELB_LB_STREAMERS_LADDIOLETDELEGATE_H

#include "lb/streamers/SimpleBounceBackDelegate.h"

namespace hemelb
{
  namespace lb
  {
    namespace streamers
    {

      template<typename CollisionImpl>
      class LaddIoletDelegate : public SimpleBounceBackDelegate<CollisionImpl>
      {
        public:
          typedef CollisionImpl CollisionType;
          typedef typename CollisionType::CKernel::LatticeType LatticeType;

          LaddIoletDelegate(CollisionType& delegatorCollider, kernels::InitParams& initParams) :
              SimpleBounceBackDelegate<CollisionType>(delegatorCollider, initParams),
                  bValues(initParams.boundaryObject)
          {
          }

#ifdef HEMELB_USE_GPU
          /** Function that returns the wall mom. for the case of Velocity BCs (LaddIolet)
              Remember that it does not take the following into account (multiply with the local density):
                  if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
                    wallMom *= hydroVars.density;
          */
          inline void Eval_wallMom(const LbmParameters* lbmParams,
                       geometry::LatticeData* const latticeData,
                       const geometry::Site<geometry::LatticeData>& site,
                       kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
                       const Direction& ii,
                       LatticeVelocity* wallMom_tobepassed)
          {
            int boundaryId = site.GetIoletId();
            iolets::InOutLetVelocity* iolet =
                dynamic_cast<iolets::InOutLetVelocity*>(bValues->GetLocalIolet(boundaryId));
            LatticePosition sitePos(site.GetGlobalSiteCoords());

            LatticePosition halfWay(sitePos);
            halfWay.x += 0.5 * LatticeType::CX[ii];
            halfWay.y += 0.5 * LatticeType::CY[ii];
            halfWay.z += 0.5 * LatticeType::CZ[ii];

            LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
            //TODO: Add site.GetGlobalSiteCoords() as a first argument?

            //printf("Density: %.5f \n\n", hydroVars.density);
            // TODO: Make sure that density value is available
            // GPU version does not have access to the hydroVars.density.
            // Use the propertyCache.densityCache.Get(site_Index)) instead.
            /*if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
            {
              wallMom *= hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
            }*/
            //printf("Entered LADDIOLET delegate!!! \n\n" ); // It does enter here.

            // Testing
            /*//if(wallMom.x >=1e-4 && wallMom.y >=1e-4 && wallMom.z >=1e-4 )
            if(wallMom.x !=0 || wallMom.y !=0 || wallMom.z !=0 )
              printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", ii, wallMom.x, wallMom.y, wallMom.z);
            */

            *wallMom_tobepassed = wallMom;
          }
#endif

          inline void StreamLink(const LbmParameters* lbmParams,
                                 geometry::LatticeData* const latticeData,
                                 const geometry::Site<geometry::LatticeData>& site,
                                 kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
                                 const Direction& ii)
          {
            // Translating from Ladd, J. Fluid Mech. "Numerical simulations
            // of particulate suspensions via a discretized Boltzmann
            // equation. Part 1. Theoretical foundation", 1994
            // Eq (3.2) -- simple bounce-back -- becomes:
            //   f_i'(r, t+1) = f_i(r, t*)
            // Eq (3.3) --- modified BB -- becomes:
            //   f_i'(r, t+1) = f_i(r, t*) - 2 a1_i \rho u . c_i
            // where u is the velocity of the boundary half way along the
            // link and a1_i = w_1 / cs2

            int boundaryId = site.GetIoletId();
            iolets::InOutLetVelocity* iolet =
                dynamic_cast<iolets::InOutLetVelocity*>(bValues->GetLocalIolet(boundaryId));
            LatticePosition sitePos(site.GetGlobalSiteCoords());

            LatticePosition halfWay(sitePos);
            halfWay.x += 0.5 * LatticeType::CX[ii];
            halfWay.y += 0.5 * LatticeType::CY[ii];
            halfWay.z += 0.5 * LatticeType::CZ[ii];

            LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
            //TODO: Add site.GetGlobalSiteCoords() as a first argument?

            if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
            {
              wallMom *= hydroVars.density;
            }

            distribn_t correction = 2. * LatticeType::EQMWEIGHTS[ii]
                * (wallMom.x * LatticeType::CX[ii] + wallMom.y * LatticeType::CY[ii]
                    + wallMom.z * LatticeType::CZ[ii]) / Cs2;

            * (latticeData->GetFNew(SimpleBounceBackDelegate<CollisionImpl>::GetBBIndex(site.GetIndex(),
                                                                                        ii))) =
                hydroVars.GetFPostCollision()[ii] - correction;
          }

        private:
          iolets::BoundaryValues* bValues;
      };

    }
  }
}

#endif /* HEMELB_LB_STREAMERS_LADDIOLETDELEGATE_H */
