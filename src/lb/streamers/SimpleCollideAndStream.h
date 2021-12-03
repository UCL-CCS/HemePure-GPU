
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_STREAMERS_SIMPLECOLLIDEANDSTREAM_H
#define HEMELB_LB_STREAMERS_SIMPLECOLLIDEANDSTREAM_H

#include "lb/streamers/BaseStreamer.h"
#include "lb/streamers/SimpleCollideAndStreamDelegate.h"
#include "lb/kernels/BaseKernel.h"
#include "lb/HFunction.h"

namespace hemelb
{
	namespace lb
	{
		namespace streamers
		{
			template<typename CollisionImpl>
				class SimpleCollideAndStream : public BaseStreamer<SimpleCollideAndStream<CollisionImpl> >
			{
				public:
					typedef CollisionImpl CollisionType;

				private:
					CollisionType collider;
					SimpleCollideAndStreamDelegate<CollisionType> bulkLinkDelegate;
					typedef typename CollisionType::CKernel::LatticeType LatticeType;

				public:
					SimpleCollideAndStream(kernels::InitParams& initParams) :
						collider(initParams), bulkLinkDelegate(collider, initParams)
				{
				}

					template<bool tDoRayTracing>
						inline void DoStreamAndCollide(const site_t firstIndex,
								const site_t siteCount,
								const LbmParameters* lbmParams,
								geometry::LatticeData* latDat,
								lb::MacroscopicPropertyCache& propertyCache)
						{
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);

								kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

								//< @todo #126 This value of tau will be updated by some kernels within the collider code (e.g. LBGKNN). It would be nicer if tau is handled in a single place.
								hydroVars.tau = lbmParams->GetTau();

								collider.CalculatePreCollision(hydroVars, site);

								collider.Collide(lbmParams, hydroVars);

								for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ii++)
								{
									bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
								}

								BaseStreamer<SimpleCollideAndStream>::template UpdateMinsAndMaxes<tDoRayTracing>(site,
										hydroVars,
										lbmParams,
										propertyCache);
							}
						}

					template<bool tDoRayTracing>
						inline void DoPostStep(const site_t iFirstIndex,
								const site_t iSiteCount,
								const LbmParameters* iLbmParams,
								geometry::LatticeData* bLatDat,
								lb::MacroscopicPropertyCache& propertyCache)
						{
						}
			};
		}
	}
}

#endif /* HEMELB_LB_STREAMERS_SIMPLECOLLIDEANDSTREAM */
