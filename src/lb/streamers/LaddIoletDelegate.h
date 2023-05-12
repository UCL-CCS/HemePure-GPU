
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

            /*
            // Debugging - Case of Vel BCs - Remove later
            //if(wallMom.x >=1e-4 && wallMom.y >=1e-4 && wallMom.z >=1e-4 )
            if (site.GetIndex()==9919 && ii==18){
              if(wallMom.x !=0 || wallMom.y !=0 || wallMom.z !=0 )
                //printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", ii, wallMom.x, wallMom.y, wallMom.z);
                // printf("Time: %d, Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", bValues->GetTimeStep(), site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, ii, wallMom.x, wallMom.y, wallMom.z);
                printf("\n/LaddIolet/ Time: %d - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", bValues->GetTimeStep(), site.GetIndex(), ii, wallMom.x, wallMom.y, wallMom.z);
            }
            */

            *wallMom_tobepassed = wallMom;
          }


          // Correction term associated with the wall momentum
          /** Function that returns the correction term (wall mom.) for the case of Velocity BCs (LaddIolet)
              Remember that it does not take the following into account (multiply with the local density):
                  if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
                    wallMom *= hydroVars.density;
          */
          inline void Eval_wallMom_correction(const LbmParameters* lbmParams,
                       geometry::LatticeData* const latticeData,
                       const geometry::Site<geometry::LatticeData>& site,
                       kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
                       const Direction& ii,
                       double* wallMom_correction_tobepassed)
          {
            int boundaryId = site.GetIoletId();
            iolets::InOutLetVelocity* iolet =
                dynamic_cast<iolets::InOutLetVelocity*>(bValues->GetLocalIolet(boundaryId));

            LatticePosition sitePos(site.GetGlobalSiteCoords());

            // Debugging - Remove later - Oct 2022
            //printf("Value of boundaryId: %d - Site index: %d \n", boundaryId, latDat->GetSite(siteIndex); );


            LatticePosition halfWay(sitePos);
            halfWay.x += 0.5 * LatticeType::CX[ii];
            halfWay.y += 0.5 * LatticeType::CY[ii];
            halfWay.z += 0.5 * LatticeType::CZ[ii];

            LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
            //TODO: Add site.GetGlobalSiteCoords() as a first argument?

            distribn_t correction = 2. * LatticeType::EQMWEIGHTS[ii]
                * (wallMom.x * LatticeType::CX[ii] + wallMom.y * LatticeType::CY[ii]
                    + wallMom.z * LatticeType::CZ[ii]) / Cs2;

            //printf("Density: %.5f \n\n", hydroVars.density);
            // TODO: Make sure that density value is available
            // GPU version does not have access to the hydroVars.density.
            // Use the propertyCache.densityCache.Get(site_Index)) instead.
            /*if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
            {
              wallMom *= hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
            }*/
            //printf("Entered LADDIOLET delegate!!! \n\n" ); // It does enter here.


            // Debugging - Case of Vel BCs - Remove later
            //if(wallMom.x >=1e-4 && wallMom.y >=1e-4 && wallMom.z >=1e-4 )
            if (site.GetIndex()==10065 && bValues->GetTimeStep()==1000) {//ii==18){ // Or site.GetIndex()==9919
              //if(wallMom.x !=0 || wallMom.y !=0 || wallMom.z !=0 )
                //printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", ii, wallMom.x, wallMom.y, wallMom.z);
                // printf("Time: %d, Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", bValues->GetTimeStep(), site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, ii, wallMom.x, wallMom.y, wallMom.z);
                printf("\n/LaddIolet/ Time: %d - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e), correction: %.5e \n", bValues->GetTimeStep(), site.GetIndex(), ii, wallMom.x, wallMom.y, wallMom.z, correction);
            }


            *wallMom_correction_tobepassed = correction;
          }

          // Prefactor Correction term associated with the wall momentum
          /** Function that returns the prefactor correction term (associated with wall mom. without the Velocity dependency) for the case of Velocity BCs (LaddIolet)
              Remember that it does not take the following into account (multiply with the local density):
                  if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
                    wallMom *= hydroVars.density;
          */
          inline void Eval_wallMom_prefactor_correction(const LbmParameters* lbmParams,
                       geometry::LatticeData* const latticeData,
                       const geometry::Site<geometry::LatticeData>& site,
                       kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
                       const Direction& ii,
                       double* wallMom__prefactor_correction_tobepassed)
          {
            int boundaryId = site.GetIoletId();
            iolets::InOutLetVelocity* iolet =
                dynamic_cast<iolets::InOutLetVelocity*>(bValues->GetLocalIolet(boundaryId));

            LatticePosition sitePos(site.GetGlobalSiteCoords());

            // Debugging - Remove later - Oct 2022
            //printf("Value of boundaryId: %d - Site index: %d \n", boundaryId, latDat->GetSite(siteIndex); );


            LatticePosition halfWay(sitePos);
            halfWay.x += 0.5 * LatticeType::CX[ii];
            halfWay.y += 0.5 * LatticeType::CY[ii];
            halfWay.z += 0.5 * LatticeType::CZ[ii];

            // If iolet->GetVelocity is used, it returns the wall momentum (3 components) which contains the velocity(t) dependency (using the velocityTable(time))
            //LatticeVelocity wallMom_prefactor(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

            LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));
            //TODO: Add site.GetGlobalSiteCoords() as a first argument?

            distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
                * (wallMom_prefactor.x * LatticeType::CX[ii] + wallMom_prefactor.y * LatticeType::CY[ii]
                    + wallMom_prefactor.z * LatticeType::CZ[ii]) / Cs2;

            //printf("Density: %.5f \n\n", hydroVars.density);
            // TODO: Make sure that density value is available
            // GPU version does not have access to the hydroVars.density.
            // Use the propertyCache.densityCache.Get(site_Index)) instead.
            /*if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
            {
              wallMom *= hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
            }*/
            //printf("Entered LADDIOLET delegate!!! \n\n" ); // It does enter here.


            // Debugging - Case of Vel BCs - Remove later
            //if(wallMom.x >=1e-4 && wallMom.y >=1e-4 && wallMom.z >=1e-4 )
            /*
            if (site.GetIndex()==28433 || site.GetIndex()==28612 ) {//ii==18){ // Or site.GetIndex()==9919
              //if(wallMom.x !=0 || wallMom.y !=0 || wallMom.z !=0 )
                //printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", ii, wallMom.x, wallMom.y, wallMom.z);
                // printf("Time: %d, Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", bValues->GetTimeStep(), site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, ii, wallMom.x, wallMom.y, wallMom.z);
                printf("\n/LaddIolet/ Time: %d - Site Id: %ld, Dir: %d, Wall Mom prefactor (x,y,z): (%.5e, %.5e, %.5e), prefactor_correction: %.5e \n", bValues->GetTimeStep(), site.GetIndex(), ii, wallMom_prefactor.x, wallMom_prefactor.y, wallMom_prefactor.z, prefactor_correction);
            }
            */

              //if(wallMom.x !=0 || wallMom.y !=0 || wallMom.z !=0 )
                //printf("Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", ii, wallMom.x, wallMom.y, wallMom.z);
                // printf("Time: %d, Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", bValues->GetTimeStep(), site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, ii, wallMom.x, wallMom.y, wallMom.z);

              /*
              if(prefactor_correction!=0)
                printf("\n/LaddIolet/ Time: %d - Site Id: %ld, Dir: %d, Wall Mom prefactor (x,y,z): (%.5e, %.5e, %.5e), prefactor_correction: %.5e \n", bValues->GetTimeStep(), site.GetIndex(), ii, wallMom_prefactor.x, wallMom_prefactor.y, wallMom_prefactor.z, prefactor_correction);
              */


            *wallMom__prefactor_correction_tobepassed = prefactor_correction;
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
