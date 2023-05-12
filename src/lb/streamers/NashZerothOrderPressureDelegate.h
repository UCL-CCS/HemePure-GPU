
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_STREAMERS_NASHZEROTHORDERPRESSUREDELEGATE_H
#define HEMELB_LB_STREAMERS_NASHZEROTHORDERPRESSUREDELEGATE_H

#include "util/utilityFunctions.h"
#include "lb/streamers/BaseStreamerDelegate.h"

namespace hemelb
{
	namespace lb
	{
		namespace streamers
		{
			template<typename CollisionImpl>
				class NashZerothOrderPressureDelegate : public BaseStreamerDelegate<CollisionImpl>
			{
				public:
					typedef CollisionImpl CollisionType;
					typedef typename CollisionType::CKernel::LatticeType LatticeType;

					NashZerothOrderPressureDelegate(CollisionType& delegatorCollider, kernels::InitParams& initParams) :
						collider(delegatorCollider), iolet(*initParams.boundaryObject)
				{
				}

					inline void StreamLink(const LbmParameters* lbmParams,
							geometry::LatticeData* const latticeData,
							const geometry::Site<geometry::LatticeData>& site,
							kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
							const Direction& direction)
					{
						int boundaryId = site.GetIoletId();

						// Set the density at the "ghost" site to be the density of the iolet.
						distribn_t ghostDensity = iolet.GetBoundaryDensity(boundaryId);

						//printf("Site Index: %lld - BoundaryId: %d - Value for ghostDensity = %.3f \n\n", ghostDensity, site.GetIndex(), boundaryId);

						// Calculate the velocity at the ghost site, as the component normal to the iolet.
						util::Vector3D<float> ioletNormal = iolet.GetLocalIolet(boundaryId)->GetNormal();

						// printf("iNLET: %d oR Outlet:%D Boundary ID: %d, Components of ioletNormal.x= %.3f, ioletNormal.y= %.3f, ioletNormal.z= %.3f \n", site.hadInlet, site.hadOutlet, boundaryId, ioletNormal.x, ioletNormal.y, ioletNormal.z);
						// printf("zCoord: %d - Density = %.5e - Boundary ID: %d, Components of ioletNormal.x= %.3f, ioletNormal.y= %.3f, ioletNormal.z= %.3f \n", site.GetGlobalSiteCoords().z, ghostDensity, boundaryId, ioletNormal.x, ioletNormal.y, ioletNormal.z);


						// Note that the division by density compensates for the fact that v_x etc have momentum
						// not velocity.
						distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

						// TODO it's ugly that we have to do this.
						// TODO having to give 0 as an argument is also ugly.
						// TODO it's ugly that we have to give hydroVars a nonsense distribution vector
						// that doesn't get used.
						kernels::HydroVars<typename CollisionType::CKernel> ghostHydrovars(site);

						ghostHydrovars.density = ghostDensity;
						ghostHydrovars.momentum = ioletNormal * component * ghostDensity;

						collider.kernel.CalculateFeq(ghostHydrovars, 0);

						Direction unstreamed = LatticeType::INVERSEDIRECTIONS[direction];

						*latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed)
							= ghostHydrovars.GetFEq()[unstreamed];
					}

					//--------------------------------------------------------------------
					inline void Eval_wallMom(const LbmParameters* lbmParams,
							geometry::LatticeData* const latticeData,
							const geometry::Site<geometry::LatticeData>& site,
							kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
								 const Direction& direction, LatticeVelocity* wallMom_tobepassed)
					{
					  printf("Entering branch in NashZerothOrderPressure from LADDIOLET \n\n");
					}
					//--------------------------------------------------------------------

					//--------------------------------------------------------------------
					inline void Eval_wallMom_correction(const LbmParameters* lbmParams,
							geometry::LatticeData* const latticeData,
							const geometry::Site<geometry::LatticeData>& site,
							kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
								 const Direction& direction, double* wallMom_correction_tobepassed)
					{
					  printf("Entering branch in NashZerothOrderPressure from LADDIOLET \n\n");
					}
					//--------------------------------------------------------------------

					//--------------------------------------------------------------------
					inline void Eval_wallMom_prefactor_correction(const LbmParameters* lbmParams,
							geometry::LatticeData* const latticeData,
							const geometry::Site<geometry::LatticeData>& site,
							kernels::HydroVars<typename CollisionType::CKernel>& hydroVars,
								 const Direction& direction, double* wallMom_correction_tobepassed)
					{
						printf("Entering branch in NashZerothOrderPressure from LADDIOLET... Check the CMake file for the BCs options \n\n");
					}
					//--------------------------------------------------------------------


				protected:
					CollisionType& collider;
					iolets::BoundaryValues& iolet;
			};
		}
	}
}

#endif // HEMELB_LB_STREAMERS_NASHZEROTHORDERPRESSUREDELEGATE_H
