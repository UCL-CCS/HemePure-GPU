
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.


#ifndef HEMELB_LB_KERNELS_LBGK_GUO_FORCING_H
#define HEMELB_LB_KERNELS_LBGK_GUO_FORCING_H

#include <cstdlib>
#include <cmath>
#include "lb/HFunction.h"
#include "util/utilityFunctions.h"
#include "lb/kernels/BaseKernel.h"

namespace hemelb
{
	namespace lb
	{
		namespace kernels
		{
			/**
			 * Implements the LBGK single-relaxation time kernel, including Guo Forcing
			 *
			 * Forcing implemented following:
			 * Phys. Rev. E 65, 046308 (2002)
			 * Zhaoli Guo, Chuguang Zheng, and Baochang Shi
			 */
			template<class LatticeType>
				class LBGKGuoForcing : public BaseKernel<LBGKGuoForcing<LatticeType>, LatticeType>
			{
				public:
					LBGKGuoForcing(InitParams& initParams) :
						BaseKernel<LBGKGuoForcing<LatticeType>, LatticeType>()
				{
				}

					// Adds forcing to momentum
					void DoCalculateDensityMomentumFeq(HydroVars<LBGKGuoForcing>& hydroVars, site_t index);
					// Forwards to LBGK base class
					void DoCalculateFeq(HydroVars<LBGKGuoForcing>&, site_t);
					// Adds forcing to collision term
					void DoCollide(const LbmParameters* const lbmParams,
							HydroVars<LBGKGuoForcing>& hydroVars);
			};

			template<class LatticeType>
				struct HydroVars<LBGKGuoForcing<LatticeType> > : HydroVarsBase<LatticeType>
				{

					friend class LBGKGuoForcing<LatticeType> ;
					public:
					// Pointer to force at this site
					const LatticeForceVector& force;

					template<class DataSource>
						HydroVars(geometry::Site<DataSource> const &_site) :
							HydroVarsBase<LatticeType>(_site), force(_site.GetForce())
					{
					}

					HydroVars(const distribn_t* const f, const LatticeForceVector& _force) :
						HydroVarsBase<LatticeType>(f), force(_force)
					{
					}

					protected:
					// Guo lattice distribution of external force contributions
					// as calculated in lattice::CalculateForceDistribution.
					FVector<LatticeType> forceDist;
				};

			template<class LatticeType>
				void LBGKGuoForcing<LatticeType>::DoCalculateDensityMomentumFeq(
						HydroVars<LBGKGuoForcing<LatticeType> >& hydroVars, site_t index)
				{
					LatticeType::CalculateDensityMomentumFEq(
							hydroVars.f,
							hydroVars.force.x,
							hydroVars.force.y,
							hydroVars.force.z,
							hydroVars.density,
							hydroVars.momentum.x,
							hydroVars.momentum.y,
							hydroVars.momentum.z,
							hydroVars.velocity.x,
							hydroVars.velocity.y,
							hydroVars.velocity.z,
							hydroVars.f_eq.f);

					//{
					//	int i = 0;
					//	char hostname[256];
					//	gethostname(hostname, sizeof(hostname));
					//	printf("PID %d on %s ready for attach\n", getpid(), hostname);
					//	fflush(stdout);
					//	while (0 == i)
					//		sleep(5);
					//}

					for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
						hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
				}

			template<class LatticeType>
				void LBGKGuoForcing<LatticeType>::DoCalculateFeq(HydroVars<LBGKGuoForcing>& hydroVars,
						site_t index)
				{
					LatticeType::CalculateFeq(hydroVars.density,
							hydroVars.momentum.x,
							hydroVars.momentum.y,
							hydroVars.momentum.z,
							hydroVars.f_eq.f);

					for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
						hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
				}

			template<class LatticeType>
				void LBGKGuoForcing<LatticeType>::DoCollide(const LbmParameters* const lbmParams,
						HydroVars<LBGKGuoForcing>& hydroVars)
				{
					LatticeType::CalculateForceDistribution(lbmParams->GetTau(),
							hydroVars.velocity.x,
							hydroVars.velocity.y,
							hydroVars.velocity.z,
							hydroVars.force.x,
							hydroVars.force.y,
							hydroVars.force.z,
							hydroVars.forceDist.f);

					for (Direction dir(0); dir < LatticeType::NUMVECTORS; ++dir)
						hydroVars.SetFPostCollision(dir,
								hydroVars.f[dir]
								+ hydroVars.f_neq.f[dir] * lbmParams->GetOmega()
								+ hydroVars.forceDist.f[dir]);
				};
		}
	}
}

#endif /* HEMELB_LB_KERNELS_LBGK_GUO_FORCING_H */
