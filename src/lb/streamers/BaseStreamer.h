// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_STREAMERS_BASESTREAMER_H
#define HEMELB_LB_STREAMERS_BASESTREAMER_H

#include <set>
#include <cmath>

#include "geometry/LatticeData.h"
#include "lb/LbmParameters.h"
#include "lb/kernels/BaseKernel.h"
#include "lb/MacroscopicPropertyCache.h"

namespace hemelb
{
	namespace lb
	{
		namespace streamers
		{

			/**
			 * BaseStreamer: inheritable base class for the streaming operator. The public interface
			 * here defines the complete interface usable by external code.
			 *  - Constructor(InitParams&)
			 *  - <bool tDoRayTracing> StreamAndCollide(const site_t, const site_t, const LbmParameters*,
			 *      geometry::LatticeData*)
			 *  - <bool tDoRayTracing> PostStep(const site_t, const site_t, const LbmParameters*,
			 *      geometry::LatticeData*)
			 *  - Reset(kernels::InitParams* init)
			 *
			 * The following must be implemented by concrete streamers (which derive from this class
			 * using the CRTP).
			 *  - typedef for CollisionType, the type of the collider operation.
			 *  - Constructor(InitParams&)
			 *  - <bool tDoRayTracing> DoStreamAndCollide(const site_t, const site_t, const LbmParameters*,
			 *      geometry::LatticeData*)
			 *  - <bool tDoRayTracing> DoPostStep(const site_t, const site_t, const LbmParameters*,
			 *      geometry::LatticeData*)
			 *  - DoReset(kernels::InitParams* init)
			 *
			 * The design is to for the streamers to be pretty dumb and for them to
			 * basically just control iteration over the sites and directions while
			 * delegating the logic of actually streaming to some other classes
			 * (e.g. BouzidiFirdaousLallemand delegates bulk link streaming to
			 * SimpleCollideAndStreamDelegate and wall link streaming to BFLDelagate,
			 * which uses SimpleBounceBackDelegate in the cases where it can't handle
			 * because two opposite links are both wall links).
			 */
			template<typename StreamerImpl>
				class BaseStreamer
				{
					public:
						template<bool tDoRayTracing>
							inline void StreamAndCollide(const site_t firstIndex,
									const site_t siteCount,
									const LbmParameters* lbmParams,
									geometry::LatticeData* latDat,
									lb::MacroscopicPropertyCache& propertyCache)
							{
								static_cast<StreamerImpl*> (this)->template DoStreamAndCollide<tDoRayTracing> (firstIndex,
										siteCount,
										lbmParams,
										latDat,
										propertyCache);
							}

						template<bool tDoRayTracing>
							inline void PostStep(const site_t firstIndex,
									const site_t siteCount,
									const LbmParameters* lbmParams,
									geometry::LatticeData* latDat,
									lb::MacroscopicPropertyCache& propertyCache)
							{
								// The template parameter is required because we're using the CRTP to call a
								// metaprogrammed method of the implementation class.
								static_cast<StreamerImpl*> (this)->template DoPostStep<tDoRayTracing> (firstIndex,
										siteCount,
										lbmParams,
										latDat,
										propertyCache);
							}

#ifdef HEMELB_USE_GPU
							template<bool tDoRayTracing>
								//inline std::vector<util::Vector3D<double> > GetWallMom(const site_t firstIndex,
								inline void GetWallMom(const site_t firstIndex,
										const site_t siteCount,
										const LbmParameters* lbmParams,
										geometry::LatticeData* latDat,
										lb::MacroscopicPropertyCache& propertyCache)
								{
									// The template parameter is required because we're using the CRTP to call a
									// metaprogrammed method of the implementation class.
									static_cast<StreamerImpl*> (this)->template DoGetWallMom<tDoRayTracing> (firstIndex,
											siteCount,
											lbmParams,
											latDat,
											propertyCache);
								}
#endif


					protected:
						template<bool tDoRayTracing, class LatticeType>
							inline static void UpdateMinsAndMaxes(const geometry::Site<geometry::LatticeData>& site,
									const kernels::HydroVarsBase<LatticeType>& hydroVars,
									const LbmParameters* lbmParams,
									lb::MacroscopicPropertyCache& propertyCache)
							{
								if (propertyCache.densityCache.RequiresRefresh())
								{
									propertyCache.densityCache.Put(site.GetIndex(), hydroVars.density);
								}

								if (propertyCache.velocityCache.RequiresRefresh())
								{
									propertyCache.velocityCache.Put(site.GetIndex(), hydroVars.velocity);
								}

								if (propertyCache.wallShearStressMagnitudeCache.RequiresRefresh())
								{
									distribn_t stress;

									if (!site.IsWall())
									{
										stress = NO_VALUE;
									}
									else
									{
										LatticeType::CalculateWallShearStressMagnitude(hydroVars.density,
												hydroVars.GetFNeq().f,
												site.GetWallNormal(),
												stress,
												lbmParams->GetStressParameter());
									}

									propertyCache.wallShearStressMagnitudeCache.Put(site.GetIndex(), stress);
								}

								if (propertyCache.vonMisesStressCache.RequiresRefresh())
								{
									distribn_t stress;
									StreamerImpl::CollisionType::CKernel::LatticeType::CalculateVonMisesStress(hydroVars.GetFNeq().f,
											stress,
											lbmParams->GetStressParameter());

									propertyCache.vonMisesStressCache.Put(site.GetIndex(), stress);
								}

								if (propertyCache.shearRateCache.RequiresRefresh())
								{
									distribn_t shear_rate =
										StreamerImpl::CollisionType::CKernel::LatticeType::CalculateShearRate(hydroVars.tau,
												hydroVars.GetFNeq().f,
												hydroVars.density);

									propertyCache.shearRateCache.Put(site.GetIndex(), shear_rate);
								}

								if (propertyCache.stressTensorCache.RequiresRefresh())
								{
									util::Matrix3D stressTensor;
									StreamerImpl::CollisionType::CKernel::LatticeType::CalculateStressTensor(hydroVars.density,
											hydroVars.tau,
											hydroVars.GetFNeq().f,
											stressTensor);

									propertyCache.stressTensorCache.Put(site.GetIndex(), stressTensor);
								}

								if (propertyCache.tractionCache.RequiresRefresh())
								{
									util::Vector3D<LatticeStress> tractionOnAPoint(0);

									/*
									 * Wall normals are only available at the sites marked as being at the domain edge.
									 * For the sites in the fluid bulk, the traction vector will be 0.
									 */
									if (site.IsWall())
									{
										LatticeType::CalculateTractionOnAPoint(hydroVars.density,
												hydroVars.tau,
												hydroVars.GetFNeq().f,
												site.GetWallNormal(),
												tractionOnAPoint);
									}

									propertyCache.tractionCache.Put(site.GetIndex(), tractionOnAPoint);
								}

								if (propertyCache.tangentialProjectionTractionCache.RequiresRefresh())
								{
									util::Vector3D<LatticeStress> tangentialProjectionTractionOnAPoint(0);

									/*
									 * Wall normals are only available at the sites marked as being at the domain edge.
									 * For the sites in the fluid bulk, the traction vector will be 0.
									 */
									if (site.IsWall())
									{
										LatticeType::CalculateTangentialProjectionTraction(hydroVars.density,
												hydroVars.tau,
												hydroVars.GetFNeq().f,
												site.GetWallNormal(),
												tangentialProjectionTractionOnAPoint);
									}

									propertyCache.tangentialProjectionTractionCache.Put(site.GetIndex(), tangentialProjectionTractionOnAPoint);
								}
							}
				};
		}
	}
}

#endif /* HEMELB_LB_STREAMERS_BASESTREAMER_H */
