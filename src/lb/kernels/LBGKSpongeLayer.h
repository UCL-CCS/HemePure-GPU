
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_KERNELS_LBGK_SPONGELAYER_H
#define HEMELB_LB_KERNELS_LBGK_SPONGELAYER_H

#include <cstdlib>
#include "util/utilityFunctions.h"
#include "lb/kernels/BaseKernel.h"

#include "cuda_kernels_def_decl/cuda_params.h"

namespace hemelb
{
	namespace lb
	{
		// Forward declaration of LBM
		template <class LatticeType>
			class LBM;

		namespace kernels
		{
			/**
			 * LBGKSpongeLayer: This class implements the LBGK single-relaxation time kernel with a viscous sponge layer.
			 */
			template<class LatticeType>
				class LBGKSpongeLayer : public BaseKernel<LBGKSpongeLayer<LatticeType>, LatticeType>
			{
				public:
					LBGKSpongeLayer(InitParams& initParams) :
						tau0(initParams.lbmParams->GetTau()), vRatio(initParams.lbmParams->ViscosityRatio),
						lifetime(initParams.lbmParams->SpongeLayerLifetime), state(initParams.state),
						Smagorinsky_cnst(initParams.lbmParams->Smagorinsky_const)
					{
						InitState(initParams);

/*#ifdef HEMELB_USE_GPU
						bool res_InitState_SpongeLayer_GPU = InitState_SpongeLayer_GPU(initParams);
						if (!res_InitState_SpongeLayer_GPU){
							printf("Error InitState_SpongeLayer_GPU - memcpy vTau H2D ... \n");
						}
						else{
							//hemelb::log::Logger::Log<hemelb::log::Debug, hemelb::log::Singleton>("LBGK Sponge Layer vTau memcpy Host to Device completed.");
							printf("InitState_SpongeLayer_GPU - memcpy vTau H2D completed \n");
						}
#endif
*/
					}

					inline void DoCalculateDensityMomentumFeq(HydroVars<LBGKSpongeLayer<LatticeType> >& hydroVars, site_t index)
					{
						LatticeType::CalculateDensityMomentumFEq(hydroVars.f,
								hydroVars.density,
								hydroVars.momentum.x,
								hydroVars.momentum.y,
								hydroVars.momentum.z,
								hydroVars.velocity.x,
								hydroVars.velocity.y,
								hydroVars.velocity.z,
								hydroVars.f_eq.f);

						for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
						{
							hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
						}

						CalculateTau(hydroVars, index);
					}

					inline void DoCalculateFeq(HydroVars<LBGKSpongeLayer>& hydroVars, site_t index)
					{
						LatticeType::CalculateFeq(hydroVars.density,
								hydroVars.momentum.x,
								hydroVars.momentum.y,
								hydroVars.momentum.z,
								hydroVars.f_eq.f);

						for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
						{
							hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
						}

						CalculateTau(hydroVars, index);
					}

					inline void DoCollide(const LbmParameters* const lbmParams, HydroVars<LBGKSpongeLayer>& hydroVars)
					{
						distribn_t omega = - 1.0 / hydroVars.tau;

						for (Direction direction = 0; direction < LatticeType::NUMVECTORS; ++direction)
						{
							hydroVars.SetFPostCollision(direction,
									hydroVars.f[direction] + hydroVars.f_neq.f[direction] * omega);
						}
					}

					/**
					* Perform necessary memory copies to the GPU at Initialisation
					* 	vTau
					*/
#ifdef HEMELB_USE_GPU
					 friend class LBM<LatticeType>;  // Declare LBM as a friend class

					 /*
					 bool InitState_SpongeLayer_GPU(const kernels::InitParams& initParams)
					 {
						 bool InitState_SpongeLayer_GPU_res = true;

						 printf("Calling InitState_SpongeLayer_GPU...\n");
						 //int myPiD = communicationNet.Rank();

						 // Number of fluid sites
						 site_t nFluid_sites = initParams.latDat->GetLocalFluidSiteCount();
						 printf("Number of fluid sites: %ld \n", nFluid_sites);

						 // Memory Size required
						 site_t MemSz = nFluid_sites * sizeof(distribn_t);

						 // Mem.copy vTau to the GPU (GPUDataAddr_vTau)
						 // 1. Allocate memory on the GPU
						 bool status = deviceMalloc((void**)&GPUDataAddr_vTau, MemSz);
						 if(!status){
							 fprintf(stderr, "GPU memory allocation vTau failed...\n");
							 InitState_SpongeLayer_GPU_res = false; return InitState_SpongeLayer_GPU_res;
						 }

						 // 2. Memory copy from host (Data_dbl_WallNormal_Edge_Type2) to Device (GPUDataAddr_WallNormal_Edge_Type2)
						 status = deviceMemcpy(GPUDataAddr_vTau, &vTau[0], MemSz, memcpyHostToDevice);
						 if(!status){
							 fprintf(stderr, "GPU memory transfer vTau Host To Device failed\n");
							 InitState_SpongeLayer_GPU_res = false; return InitState_SpongeLayer_GPU_res;
						 }

						 return 	InitState_SpongeLayer_GPU_res;
					 }
					 */

					 // Static method to access vTau, which was changed to static so that
					 // we can access it from class LBM and function Initialise_GPU())
					 static const distribn_t* GetvTau(site_t siteIndex)
					 {
						 return &vTau[siteIndex];
					 }
#endif

				private:
          			/**
           			*  Helper method to set/update member variables. Called from the constructor and Reset()
           			*
           			*  @param initParams struct used to store variables required for initialisation of various operators
           			*/
          			void InitState(const kernels::InitParams& initParams)
          			{
									//printf("Constructor - Enters InitState in LBGKSpongeLayer!!! \n\n");
            			vTau.resize(initParams.latDat->GetLocalFluidSiteCount());
									// Width of a sponge layer (in number of sites)
									const LatticeDistance width = initParams.lbmParams->SpongeLayerWidth;
									const LatticeDistance widthSq = width * width;
                    //LatticeTimeStep timeStep = state->GetTimeStep();

			            for (site_t i = 0; i < vTau.size(); i++)
            			{
										distribn_t vRatioTot = 1.0;
										const LatticeVector& siteLocation = initParams.latDat->GiveMeGlobalSiteCoords(i);

										//if (initParams.outletPositions.size()!=0 || initParams.inletPositions.size()!=0 )
										//	printf("Number of outlet/inlet positions: %ld / %ld \n", initParams.outletPositions.size(), initParams.inletPositions.size() );

										for (int j = 0; j < initParams.outletPositions.size(); j++)
										{
											const LatticeDistance distSq = (siteLocation - initParams.outletPositions[j]).GetMagnitudeSquared();

											// const int dist = (siteLocation - initParams.outletPositions[j]).GetByDirection(util::Direction::Direction::X);
											// const LatticeDistance distSq = dist * dist;
											const LatticeDistance dist = std::sqrt(distSq);
											//
											// Debugging - Testing
											//if (i==35933) printf("Outlet Case: %d, Site: %lu Coords:[%ld, %ld, %ld] - distSq: %f  - vRatio: %f - dist: %f, width: %f \n", j, i, siteLocation.x, siteLocation.y, siteLocation.z, distSq, vRatio, dist, width);
											//

											if (distSq <= widthSq)
											{
												// Quadratic function
												vRatioTot *= 1.0 + (vRatio - 1.0) * (dist / width - 1.0) * (dist / width - 1.0);

												// Sinusoidal function
												//vRatioTot *= (0.5 * (vRatio - 1.0)) * (1.0 + cos((PI / widthSq) * distSq)) + 1.0;
											}
										}
										// Note that viscosity is proportional to (tau - 0.5)
										vTau[i] = vRatioTot * (tau0 - 0.5) + 0.5;


										// if(timeStep > 92800){
										for (int j = 0; j < initParams.inletPositions.size(); j++)
										{
											if(j != 6 ){
												if(j == 8 || j == 22 || j == 23){
													// std::cout<<"inlet: "<<j<<std::endl;
													const LatticeDistance distSq = (siteLocation - initParams.inletPositions[j]).GetMagnitudeSquared();

													// const int dist = (siteLocation - initParams.outletPositions[j]).GetByDirection(util::Direction::Direction::X);
													// const LatticeDistance distSq = dist * dist;
													const LatticeDistance dist = std::sqrt(distSq);
													if (distSq <= widthSq)
													{
														// Quadratic function
														vRatioTot *= 1.0 + (vRatio - 1.0) * (dist / width - 1.0) * (dist / width - 1.0);

														// Sinusoidal function
														//vRatioTot *= (0.5 * (vRatio - 1.0)) * (1.0 + cos((PI / widthSq) * distSq)) + 1.0;
													}
												}
												else
												{
													const LatticeDistance distSq = (siteLocation - initParams.inletPositions[j]).GetMagnitudeSquared();

													// const int dist = (siteLocation - initParams.outletPositions[j]).GetByDirection(util::Direction::Direction::X);
													// const LatticeDistance distSq = dist * dist;
													const LatticeDistance dist = std::sqrt(distSq);
													//
													// Debugging - Testing
													//if (i==35933) printf("Inlet Case: %d, Site: %lu Coords:[%ld, %ld, %ld] - distSq: %f  - vRatio: %f - dist: %f, width: %f \n", j, i, siteLocation.x, siteLocation.y, siteLocation.z, distSq, vRatio, dist, width);
													//
													if (distSq <= 784)
													{
														// Quadratic function
														vRatioTot *= 1.0 + (vRatio - 1.0) * (dist / width - 1.0) * (dist / width - 1.0);

														// Sinusoidal function
														//vRatioTot *= (0.5 * (vRatio - 1.0)) * (1.0 + cos((PI / widthSq) * distSq)) + 1.0;
													}
												}
											}

										}
										vTau[i] = vRatioTot * (tau0 - 0.5) + 0.5;
                    // }
										//if (i==35933) printf("Site: %lu - vRatio: %f - vTau: %.3e \n", i, vRatio, vTau[i]);

            			}
          			}

					/**
					* Calculate the relaxation time of the collision and temporaily store it in hydroVars.
					* The sponge layer is maintained for a certain time and then dissolved.
					*/
					inline void CalculateTau(HydroVars<LBGKSpongeLayer<LatticeType> >& hydroVars, site_t index)
					{
						LatticeTimeStep timeStep = state->GetTimeStep();
						double tau_les = compute_tau_smagorinsky(hydroVars);

						// printf("tau_les: %lf\n", tau_les);
						if (timeStep <= lifetime / 2)
						{
							if(vTau[index] == tau0){
								hydroVars.tau = tau_les;
							}else{
								hydroVars.tau = vTau[index];
							}
						}
						else if (timeStep < lifetime)
						{
							// Linear decay from vTau to tau0
							// hydroVars.tau = (tau0 - vTau[index]) * 2.0 / lifetime * timeStep + (2.0 * vTau[index] - tau0);
							hydroVars.tau = (tau_les - vTau[index]) * 2.0 / lifetime * timeStep + (2.0 * vTau[index] - tau_les);
						}
						else
						{
							// hydroVars.tau = tau0;
							hydroVars.tau = tau_les;
						}
					}

					double compute_tau_smagorinsky(HydroVars<LBGKSpongeLayer<LatticeType> >& hydroVars) const
					{
						// calculate tau using smagorinsky local correction
						double dx = 1.0;
						double dt = 1.0;
						double C = dx / dt;
						double rho1 = 1.0;
						double localTau;
						double C_smag = Smagorinsky_cnst;//0.1; // Value provided from the input file
						printf("From LES Sponge - Smagorinsky const: %.3f  \n\n", Smagorinsky_cnst);

						// Compute non-equilibrium values
						for (unsigned int ii = 0; ii < LatticeType::NUMVECTORS; ++ii)
						{
							hydroVars.f_neq.f[ii] = hydroVars.f[ii] - hydroVars.f_eq.f[ii];
						}
						double Q_12 = 0.0;
						// Calculate diagonal and upper diagonal of the non equilibrium stress tensor
						for (int i = 0; i < 3; ++i) {
							for (int j = 0; j < 3; ++j) {
								double qij = 0.0;
								for (int v = 0; v < LatticeType::NUMVECTORS; ++v) {
									qij += LatticeType::discreteVelocityVectors[i][v] * LatticeType::discreteVelocityVectors[j][v] * hydroVars.f_neq.f[v];
								}
								Q_12 += qij * qij;
							}
						}
						Q_12 = sqrt(Q_12);
						// eq 36 Koda 2015, csmag is smagorinsky constant here is c_smag^2 in the paper is c_smag
						localTau = 1. / 2. *
									(tau0 + sqrt((tau0 * rho1 * C)*(tau0 * rho1 * C) +
									18.0 * 1.4142135623730950488016887242097 * rho1 * C_smag * C_smag * Q_12) / (rho1 * C));

						return localTau;
					}

					// Normal relaxation time
					const distribn_t tau0;
					// Ratio of the maximum viscosity in the sponge layer to the normal viscosity
					const Dimensionless vRatio;
					// Lifetime of the sponge layer
					const LatticeTimeStep lifetime;

					// Smagorinsky constant
					const Dimensionless Smagorinsky_cnst;

					// Pointer to the simulation state which provides the current time step.
					SimulationState* state;

					// Vector containing the viscous relaxation time for each site in the domain.
          static std::vector<distribn_t> vTau;

#ifdef HEMELB_USE_GPU
					void *GPUDataAddr_vTau;
#endif

			};

			// Define the static member outside the class definition
			template<class LatticeType>
			std::vector<distribn_t> LBGKSpongeLayer<LatticeType>::vTau;

		}
	}
}

#endif /* HEMELB_LB_KERNELS_LBGK_SPONGELAYER_H */
