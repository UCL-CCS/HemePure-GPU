
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_STREAMERS_STREAMERTYPEFACTORY_H
#define HEMELB_LB_STREAMERS_STREAMERTYPEFACTORY_H

#include "lb/kernels/BaseKernel.h"
#include "lb/streamers/BaseStreamer.h"
#include "lb/streamers/SimpleCollideAndStreamDelegate.h"

namespace hemelb
{
	namespace lb
	{
		namespace streamers
		{
			/**
			 * Template to produce Streamers that can cope with fluid-fluid and
			 * fluid-wall links. Requires two classes as arguments: 1) the Collision
			 * class and 2) a StreamerDelegate class that will handle the wall links.
			 *
			 * It is intended that a simpler metafunction partially specialise this
			 * template on WallLinkImpl.
			 */
			template<typename CollisionImpl, typename WallLinkImpl>
				class WallStreamerTypeFactory : public BaseStreamer<WallStreamerTypeFactory<CollisionImpl, WallLinkImpl> >
			{
				public:
					typedef CollisionImpl CollisionType;

				private:
					CollisionType collider;
					SimpleCollideAndStreamDelegate<CollisionType> bulkLinkDelegate;
					WallLinkImpl wallLinkDelegate;

					typedef typename CollisionType::CKernel::LatticeType LatticeType;

				public:
					WallStreamerTypeFactory(kernels::InitParams& initParams) :
						collider(initParams), bulkLinkDelegate(collider, initParams), wallLinkDelegate(collider, initParams)
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

								//const distribn_t* fOld = site.GetFOld<LatticeType> ();

								kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

								///< @todo #126 This value of tau will be updated by some kernels within the collider code (e.g. LBGKNN). It would be nicer if tau is handled in a single place.
								hydroVars.tau = lbmParams->GetTau();

								collider.CalculatePreCollision(hydroVars, site);

								collider.Collide(lbmParams, hydroVars);

								for (Direction ii = 0; ii < LatticeType::NUMVECTORS; ii++)
								{
									if (site.HasWall(ii))
									{
										wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
									else
									{
										bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
								}

								//TODO: Necessary to specify sub-class?
								BaseStreamer<WallStreamerTypeFactory>::template UpdateMinsAndMaxes<tDoRayTracing>(site,
										hydroVars,
										lbmParams,
										propertyCache);
							}
						}

					template<bool tDoRayTracing>
						inline void DoPostStep(const site_t firstIndex,
								const site_t siteCount,
								const LbmParameters* lbmParameters,
								geometry::LatticeData* latticeData,
								lb::MacroscopicPropertyCache& propertyCache)
						{
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								geometry::Site<geometry::LatticeData> site = latticeData->GetSite(siteIndex);
								for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
								{
									if (site.HasWall(direction))
									{
										wallLinkDelegate.PostStepLink(latticeData, site, direction);
									}
								}
							}
						}
			};

			/**
			 * Template to produce Streamers that can cope with fluid-fluid and
			 * fluid-iolet links. Requires two classes as arguments: 1) the Collision
			 * class and 2) a StreamerDelegate class that will handle the iolet links.
			 *
			 * It is intended that a simpler metafunction partially specialise this
			 * template on IoletLinkImpl.
			 */
			template<typename CollisionImpl, typename IoletLinkImpl>
				class IoletStreamerTypeFactory : public BaseStreamer<IoletStreamerTypeFactory<CollisionImpl, IoletLinkImpl> >
			{
				public:
					typedef CollisionImpl CollisionType;
					std::vector<util::Vector3D<double> > wallMom_Vect3D;

				private:
					CollisionType collider;
					SimpleCollideAndStreamDelegate<CollisionType> bulkLinkDelegate;
					IoletLinkImpl ioletLinkDelegate;

					typedef typename CollisionType::CKernel::LatticeType LatticeType;

				public:
					IoletStreamerTypeFactory(kernels::InitParams& initParams) :
						collider(initParams), bulkLinkDelegate(collider, initParams), ioletLinkDelegate(collider, initParams)
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

								//const distribn_t* fOld = site.GetFOld<LatticeType> ();

								kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

								///< @todo #126 This value of tau will be updated by some kernels within the collider code (e.g. LBGKNN). It would be nicer if tau is handled in a single place.
								hydroVars.tau = lbmParams->GetTau();

								collider.CalculatePreCollision(hydroVars, site);

								collider.Collide(lbmParams, hydroVars);

								for (Direction ii = 0; ii < LatticeType::NUMVECTORS; ii++)
								{
									if (site.HasIolet(ii))
									{
										ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
									else
									{
										bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
								}

								//TODO: Necessary to specify sub-class?
								BaseStreamer<IoletStreamerTypeFactory>::template UpdateMinsAndMaxes<tDoRayTracing>(site,
										hydroVars,
										lbmParams,
										propertyCache);
							}
						}

					template<bool tDoRayTracing>
						inline void DoPostStep(const site_t firstIndex,
								const site_t siteCount,
								const LbmParameters* lbmParameters,
								geometry::LatticeData* latticeData,
								lb::MacroscopicPropertyCache& propertyCache)
						{
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								geometry::Site<geometry::LatticeData> site = latticeData->GetSite(siteIndex);
								for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
								{
									if (site.HasIolet(direction))
									{
										ioletLinkDelegate.PostStepLink(latticeData, site, direction);
									}
								}
							}
						}

//------------------------------------------------------------------------------
#ifdef HEMELB_USE_GPU

					// Get the Wall momemtum term (3 components: x,y,z) - to be passed to the GPU
					// Case of Iolet collision-streaming type - Vel BCs (LADDIOLET)
					template<bool tDoRayTracing>
						//inline std::vector<util::Vector3D<double> > DoGetWallMom(const site_t firstIndex,
						inline void DoGetWallMom(const site_t firstIndex,
												const site_t siteCount,
												const LbmParameters* lbmParams,
												geometry::LatticeData* latDat,
												lb::MacroscopicPropertyCache& propertyCache)
						{

							//std::vector<util::Vector3D<double> > wallMom_Vect3D;

							LatticeVelocity wallMom_received; // typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
							/*std::vector<double> wallMom_Vect_x;
							std::vector<double> wallMom_Vect_y;
							std::vector<double> wallMom_Vect_z;
							*/

							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
								kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

								for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
								{
									if (site.HasIolet(direction))
									{
										ioletLinkDelegate.Eval_wallMom(lbmParams, latDat, site, hydroVars, direction, &wallMom_received);
										// printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
									}
									else
									{
										wallMom_received.x = 0.0; wallMom_received.y = 0.0; wallMom_received.z = 0.0;
									}

									/*
									if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
			            {
			              wallMom_received *= 1.0; //propertyCache.densityCache.Get(siteIndex); //hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
										printf("Loc.1 : Entering the loop for IsLatticeCompressible! Density = %.5f \n", propertyCache.densityCache.Get(siteIndex) );
										//if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
										//	printf("Loc:1, Dir: %d, Density: %.5f, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, propertyCache.densityCache.Get(siteIndex), wallMom_received.x, wallMom_received.y, wallMom_received.z);
									}
									*/
									/* // Testing - Print the density from the propertyCache
									printf("Loc.1 - Density from densityCache: %.5f \n\n", propertyCache.densityCache.Get(siteIndex));

									// Testing - print the values received
									if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
										printf("Loc:1, Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
									*/

									/*
									// Debugging - Case of Vel BCs - Remove later
									if (site.GetIndex()==9919 && direction==18){
										if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
											printf("IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																	direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);

											//printf("Loc.1 / Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, \
																																		direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
									}
									*/

										// Write to propertyCache starting from location 0
									 	//propertyCache.wallMom_Cache.Put((siteIndex - firstIndex)*LatticeType::NUMVECTORS + direction, wallMom_received);

										// Write to propertyCache starting from the location based on the actual fluid ID
									 	propertyCache.wallMom_Cache.Put(siteIndex*LatticeType::NUMVECTORS + direction, wallMom_received);

										/*
										//----------------------------------------------------------
										// Passed the test: Test that I read the correct values back - Remove later
										LatticeVelocity site_WallMom_read = propertyCache.wallMom_Cache.Get(siteIndex*LatticeType::NUMVECTORS + direction);

										if (site.GetIndex()==9919 && direction==18){
											if(site_WallMom_read.x !=0 || site_WallMom_read.y !=0 || site_WallMom_read.z !=0)
											printf("Read from Cache IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																direction, site_WallMom_read.x, site_WallMom_read.y, site_WallMom_read.z);
									  }
										//----------------------------------------------------------
										*/
									}
								}

								//return wallMom_Vect3D;
							} // End of inline void DoGetWallMom


							//
							// Get the correction term associated with the Wall momemtum term (single term value) - to be passed to the GPU
							// Case of (A) Iolet collision-streaming type - Vel BCs (LADDIOLET)
							// TODO: Actually hydrovars is not used by Eval_wallMom_correction - can be removed from the arguments' list
							template<bool tDoRayTracing>
								inline void DoGetWallMom_correction(const site_t firstIndex,
														const site_t siteCount,
														const LbmParameters* lbmParams,
														geometry::LatticeData* latDat,
														lb::MacroscopicPropertyCache& propertyCache)
								{

									// This needs to change - The type should be just a single array
									// TODO
									// LatticeVelocity wallMom_received;	// typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
									//std::vector<double> wallMom_correction_received;
									double wallMom_correction_received;

									for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
									{
										geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
										kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

										for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
										{
											if (site.HasIolet(direction))
											{
												ioletLinkDelegate.Eval_wallMom_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_correction_received);
												// printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
											}
											else
											{
												wallMom_correction_received = 0.0;
											}

											/*
											if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
					            {
					              wallMom_received *= 1.0; //propertyCache.densityCache.Get(siteIndex); //hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
												printf("Loc.1 : Entering the loop for IsLatticeCompressible! Density = %.5f \n", propertyCache.densityCache.Get(siteIndex) );
												//if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
												//	printf("Loc:1, Dir: %d, Density: %.5f, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, propertyCache.densityCache.Get(siteIndex), wallMom_received.x, wallMom_received.y, wallMom_received.z);
											}
											*/
											/* // Testing - Print the density from the propertyCache
											printf("Loc.1 - Density from densityCache: %.5f \n\n", propertyCache.densityCache.Get(siteIndex));

											// Testing - print the values received
											if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
												printf("Loc:1, Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
											*/

											/*
											// Debugging - Case of Vel BCs - Remove later
											if (site.GetIndex()==9919 && direction==18){
												if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
													printf("IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																			direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);

													//printf("Loc.1 / Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, \
																																				direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
											}
											*/

												// Write to propertyCache starting from location 0
											 	// propertyCache.wallMom_Cache.Put((siteIndex - firstIndex)*LatticeType::NUMVECTORS + direction, wallMom_received);

												// TODO: Check the following for the correction term
												// Write to propertyCache starting from the location based on the actual fluid ID
											 	propertyCache.wallMom_correction_Cache.Put(siteIndex*LatticeType::NUMVECTORS + direction, wallMom_correction_received);

												/*
												//----------------------------------------------------------
												// Passed the test: Test that I read the correct values back - Remove later
												LatticeVelocity site_WallMom_read = propertyCache.wallMom_Cache.Get(siteIndex*LatticeType::NUMVECTORS + direction);

												if (site.GetIndex()==9919 && direction==18){
													if(site_WallMom_read.x !=0 || site_WallMom_read.y !=0 || site_WallMom_read.z !=0)
													printf("Read from Cache IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																		direction, site_WallMom_read.x, site_WallMom_read.y, site_WallMom_read.z);
											  }
												//----------------------------------------------------------
												*/
											}
										}

									} // End of inline void DoGetWallMom_correction



									//
									// Get the correction term associated with the Wall momemtum term (single term value) - to be passed to the GPU
									// Case of (A) Iolet collision-streaming type - Vel BCs (LADDIOLET)
									// TODO: Actually hydrovars is not used by Eval_wallMom_correction - can be removed from the arguments' list
									template<bool tDoRayTracing>
										//inline std::vector<distribn_t> DoGetWallMom_correction_Direct(const site_t firstIndex,
										inline void DoGetWallMom_correction_Direct(const site_t firstIndex,
																const site_t siteCount,
																const LbmParameters* lbmParams,
																geometry::LatticeData* latDat,
																lb::MacroscopicPropertyCache& propertyCache,
																std::vector<double>& wallMom_correction_Iolet)
										{
											//std::vector<distribn_t> result_wallMom_correction_Direct_tobePassed;

											std::vector<distribn_t> result_wallMom_correction_Direct;
											result_wallMom_correction_Direct.reserve(siteCount*LatticeType::NUMVECTORS);

											// This needs to change - The type should be just a single array
											// TODO
											// LatticeVelocity wallMom_received;	// typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
											//std::vector<double> wallMom_correction_received;
											double wallMom_correction_received;

											for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
											{
												geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
												kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

												for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
												{
													if (site.HasIolet(direction))
													{
														ioletLinkDelegate.Eval_wallMom_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_correction_received);
														//printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
													}
													else
													{
														wallMom_correction_received = 0.0;
													}

													// Debugging - Case of Vel BCs - Remove later
													/*if (site.GetIndex()==9919 && direction==5){
														if(wallMom_correction_received !=0 )
															printf("IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom Correction: %.5e \n", siteIndex, \
																																					direction, wallMom_correction_received);

															//printf("Loc.1 / Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, \
																																						direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
													}*/


													result_wallMom_correction_Direct.push_back(wallMom_correction_received);

												}
											}

											/*
											std::vector<double>::iterator it;

									  	// Print the values:
									  	std::cout << "result_wallMom_correction_Direct contains:";
									  	for (it=result_wallMom_correction_Direct.begin(); it!=result_wallMom_correction_Direct.end(); ++it)
									      	std::cout << ' ' << *it;
									  			std::cout << '\n';
													*/

													//result_wallMom_correction_Direct_tobePassed = result_wallMom_correction_Direct;
													//return result_wallMom_correction_Direct_tobePassed;

													// Check whether it makes a difference without the line below - i.e. work directly with wallMom_correction_Iolet from the beginning
													wallMom_correction_Iolet = result_wallMom_correction_Direct;

										} // End of inline void DoGetWallMom_correction_Direct


										//
										// Get the prefactor associated with the correction term associated with the Wall momemtum term (single term value) - to be passed to the GPU
										// Case of (A) Iolet collision-streaming type - Vel BCs (LADDIOLET)
										// TODO: Actually hydrovars is not used by Eval_wallMom_prefactor_correction - can be removed from the arguments' list
										template<bool tDoRayTracing>
											inline void DoGetWallMom_prefactor_correction_Direct(const site_t firstIndex,
																	const site_t siteCount,
																	const LbmParameters* lbmParams,
																	geometry::LatticeData* latDat,
																	lb::MacroscopicPropertyCache& propertyCache,
																	std::vector<double>& wallMom_prefactor_correction_Iolet)
											{

												std::vector<distribn_t> result_wallMom_prefactor_correction_Direct;
												result_wallMom_prefactor_correction_Direct.reserve(siteCount*LatticeType::NUMVECTORS);

												// This needs to change - The type should be just a single array
												// TODO
												// LatticeVelocity wallMom_received;	// typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
												//std::vector<double> wallMom_correction_received;
												double wallMom_prefactor_correction_received;

												for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
												{
													geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
													kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

													for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
													{
														if (site.HasIolet(direction))
														{
															ioletLinkDelegate.Eval_wallMom_prefactor_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_prefactor_correction_received);
															//printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
														}
														else
														{
															wallMom_prefactor_correction_received = 0.0;
														}

														// Debugging - Case of Vel BCs - Remove later
														/*if (site.GetIndex()==9919 && direction==5){
															if(wallMom_correction_received !=0 )
																printf("IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom Correction: %.5e \n", siteIndex, \
																																						direction, wallMom_correction_received);

																//printf("Loc.1 / Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, \
																																							direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
														}*/
														/*
														if(siteIndex==28433 || siteIndex ==28612){
															if(wallMom_prefactor_correction_received!=0 ){
																printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
																printf("/StreamerTypeFactory - LaddIolet/ - Site Id: %ld, Dir: %d, prefactor_correction received: %.5e \n\n", site.GetIndex(), direction, wallMom_prefactor_correction_received);
															}
														}
														*/
														result_wallMom_prefactor_correction_Direct.push_back(wallMom_prefactor_correction_received);

													}
												}

												/*
												std::vector<double>::iterator it;

										  	// Print the values:
										  	std::cout << "result_wallMom_prefactor_correction_Direct contains:";
										  	for (it=result_wallMom_prefactor_correction_Direct.begin(); it!=result_wallMom_prefactor_correction_Direct.end(); ++it)
										      	std::cout << ' ' << *it;
										  			std::cout << '\n';
														*/

														// Check whether it makes a difference without the line below - i.e. work directly with wallMom_correction_Iolet from the beginning
														wallMom_prefactor_correction_Iolet = result_wallMom_prefactor_correction_Direct;

											} // End of inline void DoGetWallMom_prefactor_correction_Direct

#endif
//------------------------------------------------------------------------------
			};

			/**
			 * Template to produce Streamers that can cope with fluid-fluid,
			 * fluid-wall and fluid-iolet links. Requires three classes as arguments:
			 * 1) the Collision class,
			 * 2) a StreamerDelegate class that will handle the wall links, and
			 * 3) a StreamerDelegate class that will handle the iolet links.
			 *
			 * It is intended that a simpler metafunction partially specialise this
			 * template on WallLinkImpl and IoletLinkImpl.
			 */
			template<typename CollisionImpl, typename WallLinkImpl, typename IoletLinkImpl>
				class WallIoletStreamerTypeFactory : public BaseStreamer<WallIoletStreamerTypeFactory<CollisionImpl,
				WallLinkImpl, IoletLinkImpl> >
			{
				public:
					typedef CollisionImpl CollisionType;
					typedef typename CollisionType::CKernel::LatticeType LatticeType;
					std::vector<util::Vector3D<double> > wallMom_Vect3D;

				private:
					CollisionType collider;
					SimpleCollideAndStreamDelegate<CollisionType> bulkLinkDelegate;
					WallLinkImpl wallLinkDelegate;
					IoletLinkImpl ioletLinkDelegate;

				public:
					WallIoletStreamerTypeFactory(kernels::InitParams& initParams) :
						collider(initParams), bulkLinkDelegate(collider, initParams), wallLinkDelegate(collider, initParams),
						ioletLinkDelegate(collider, initParams)
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

								//const distribn_t* fOld = site.GetFOld<LatticeType> ();

								kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

								///< @todo #126 This value of tau will be updated by some kernels within the collider code (e.g. LBGKNN). It would be nicer if tau is handled in a single place.
								hydroVars.tau = lbmParams->GetTau();

								collider.CalculatePreCollision(hydroVars, site);

								collider.Collide(lbmParams, hydroVars);

								for (Direction ii = 0; ii < LatticeType::NUMVECTORS; ii++)
								{
									if (site.HasIolet(ii))
									{
										ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
									else if (site.HasWall(ii))
									{
										wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
									else
									{
										bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
									}
								}

								//TODO: Necessary to specify sub-class?
								BaseStreamer<WallIoletStreamerTypeFactory>::template UpdateMinsAndMaxes<tDoRayTracing>(site,
										hydroVars,
										lbmParams,
										propertyCache);
							}
						}

					template<bool tDoRayTracing>
						inline void DoPostStep(const site_t firstIndex,
								const site_t siteCount,
								const LbmParameters* lbmParams,
								geometry::LatticeData* latticeData,
								lb::MacroscopicPropertyCache& propertyCache)
						{
							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								geometry::Site<geometry::LatticeData> site = latticeData->GetSite(siteIndex);
								for (unsigned int direction = 0; direction < LatticeType::NUMVECTORS; direction++)
								{
									if (site.HasWall(direction))
									{
										wallLinkDelegate.PostStepLink(latticeData, site, direction);
									}
									else if (site.HasIolet(direction))
									{
										ioletLinkDelegate.PostStepLink(latticeData, site, direction);
									}
								}
							}
						}

#ifdef HEMELB_USE_GPU

					// Get the Wall momemtum term (3 components: x,y,z) - to be passed to the GPU
					// Case of Iolet with Walls collision-streaming type - Vel BCs (LADDIOLET)
					template<bool tDoRayTracing>
						//inline std::vector<util::Vector3D<double> > DoGetWallMom(const site_t firstIndex,
						inline void DoGetWallMom(const site_t firstIndex,
												const site_t siteCount,
												const LbmParameters* lbmParams,
												geometry::LatticeData* latDat,
												lb::MacroscopicPropertyCache& propertyCache)
						{
							//std::vector<util::Vector3D<double> > wallMom_Vect3D;

							LatticeVelocity wallMom_received; // typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
							/*std::vector<double> wallMom_Vect_x;
							std::vector<double> wallMom_Vect_y;
							std::vector<double> wallMom_Vect_z;
							*/

							for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
							{
								geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
								kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

								for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
								{
									if (site.HasIolet(direction))
									{
										ioletLinkDelegate.Eval_wallMom(lbmParams, latDat, site, hydroVars, direction, &wallMom_received);
										// printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
									}
									else
									{
										wallMom_received.x = 0.0; wallMom_received.y = 0.0; wallMom_received.z = 0.0;
									}

									/*
									if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
			            {
			              wallMom_received *= 1.0; //propertyCache.densityCache.Get(siteIndex); //hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
										printf("Loc.2 - Entering the loop for IsLatticeCompressible! Density = %.5f \n", propertyCache.densityCache.Get(siteIndex) );
										//if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
										//	printf("Loc:2, Dir: %d, Density: %.5f, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, propertyCache.densityCache.Get(siteIndex), wallMom_received.x, wallMom_received.y, wallMom_received.z);

									}
									*/
									/*// Testing - Print the density from the propertyCache
									printf("Loc.2 - Density from densityCache: %.5f \n\n", propertyCache.densityCache.Get(siteIndex));
									*/

									/*
									// Debugging - Case of Vel BCs - Remove later
									// Testing - print the values received
									if (site.GetIndex()==9919 && direction==18){
									if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
									printf("WallIoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																	direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
										//printf("Loc.2 / Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, \
																																		direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
										//printf("Time: %d, Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", bValues->GetTimeStep(), site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, ii, wallMom.x, wallMom.y, wallMom.z);
									}
									*/

									// Write to propertyCache starting from location 0
									//propertyCache.wallMom_Cache.Put((siteIndex - firstIndex)*LatticeType::NUMVECTORS + direction, wallMom_received);

									// Write to propertyCache starting from the location based on the actual fluid ID
									propertyCache.wallMom_Cache.Put(siteIndex*LatticeType::NUMVECTORS + direction, wallMom_received);

									}
								}

								//return wallMom_Vect3D;
						} // End of function inline void DoGetWallMom


						//
						// Get the correction term associated with the Wall momemtum term (single term value) - to be passed to the GPU
						// Case of (B) Iolets with Walls collision-streaming type - Vel BCs (LADDIOLET)
						// TODO: Actually hydrovars is not used by Eval_wallMom_correction - can be removed from the arguments' list
						template<bool tDoRayTracing>
							inline void DoGetWallMom_correction(const site_t firstIndex,
													const site_t siteCount,
													const LbmParameters* lbmParams,
													geometry::LatticeData* latDat,
													lb::MacroscopicPropertyCache& propertyCache)
							{

								// This needs to change - The type should be just a single array
								// TODO
								// LatticeVelocity wallMom_received;	// typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
								//std::vector<double> wallMom_correction_received;
								double wallMom_correction_received;

								for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
								{
									geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
									kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

									for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
									{
										if (site.HasIolet(direction))
										{
											ioletLinkDelegate.Eval_wallMom_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_correction_received);
											// printf("Entering Branch StreamerTypeFactory 1 \n"); // Yes... Enters this path
										}
										else
										{
											wallMom_correction_received = 0.0;
										}

										/*
										if (CollisionType::CKernel::LatticeType::IsLatticeCompressible())
										{
											wallMom_received *= 1.0; //propertyCache.densityCache.Get(siteIndex); //hydroVars.density; // CAREFULL: density is a parameter on the host. It has to be updated at every step for this to work!!!
											printf("Loc.1 : Entering the loop for IsLatticeCompressible! Density = %.5f \n", propertyCache.densityCache.Get(siteIndex) );
											//if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
											//	printf("Loc:1, Dir: %d, Density: %.5f, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, propertyCache.densityCache.Get(siteIndex), wallMom_received.x, wallMom_received.y, wallMom_received.z);
										}
										*/
										/* // Testing - Print the density from the propertyCache
										printf("Loc.1 - Density from densityCache: %.5f \n\n", propertyCache.densityCache.Get(siteIndex));

										// Testing - print the values received
										if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
											printf("Loc:1, Dir: %d, Wall Mom_x: %.5e, Wall Mom_y: %.5e, Wall Mom_z: %.5e \n", direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
										*/

										/*
										// Debugging - Case of Vel BCs - Remove later
										if (site.GetIndex()==9919 && direction==18){
											if(wallMom_received.x !=0 || wallMom_received.y !=0 || wallMom_received.z !=0)
												printf("IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																		direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);

												//printf("Loc.1 / Coords(x,y,z): (%ld, %ld, %ld), Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", site.GetGlobalSiteCoords().x, site.GetGlobalSiteCoords().y, site.GetGlobalSiteCoords().z, \
																																			direction, wallMom_received.x, wallMom_received.y, wallMom_received.z);
										}
										*/

											// Write to propertyCache starting from location 0
											// propertyCache.wallMom_Cache.Put((siteIndex - firstIndex)*LatticeType::NUMVECTORS + direction, wallMom_received);

											// TODO: Check the following for the correction term
											// Write to propertyCache starting from the location based on the actual fluid ID
											propertyCache.wallMom_correction_Cache.Put(siteIndex*LatticeType::NUMVECTORS + direction, wallMom_correction_received);

											/*
											//----------------------------------------------------------
											// Passed the test: Test that I read the correct values back - Remove later
											LatticeVelocity site_WallMom_read = propertyCache.wallMom_Cache.Get(siteIndex*LatticeType::NUMVECTORS + direction);

											if (site.GetIndex()==9919 && direction==18){
												if(site_WallMom_read.x !=0 || site_WallMom_read.y !=0 || site_WallMom_read.z !=0)
												printf("Read from Cache IoletStreamerTypeFactory - Site Id: %ld, Dir: %d, Wall Mom (x,y,z): (%.5e, %.5e, %.5e) \n", siteIndex, \
																																	direction, site_WallMom_read.x, site_WallMom_read.y, site_WallMom_read.z);
											}
											//----------------------------------------------------------
											*/
										}
									}

									//return wallMom_Vect3D;
								} // End of inline void DoGetWallMom_correction



								//
								// Get the correction term associated with the Wall momemtum term (single term value) - to be passed to the GPU
								// Case of (B) Iolets with Walls collision-streaming type - Vel BCs (LADDIOLET)
								// TODO: Actually hydrovars is not used by Eval_wallMom_correction - can be removed from the arguments' list
								template<bool tDoRayTracing>
									//inline std::vector<distribn_t> DoGetWallMom_correction_Direct(const site_t firstIndex,
									inline void DoGetWallMom_correction_Direct(const site_t firstIndex,
															const site_t siteCount,
															const LbmParameters* lbmParams,
															geometry::LatticeData* latDat,
															lb::MacroscopicPropertyCache& propertyCache,
															std::vector<double>& wallMom_correction_Iolet)
									{
										/*
										std::vector<distribn_t> result_wallMom_correction_Direct_tobePassed;
										result_wallMom_correction_Direct_tobePassed.reserve(siteCount*LatticeType::NUMVECTORS);
										*/

										std::vector<distribn_t> result_wallMom_correction_Direct;
										result_wallMom_correction_Direct.reserve(siteCount*LatticeType::NUMVECTORS);

										// This needs to change - The type should be just a single array
										// TODO
										// LatticeVelocity wallMom_received;	// typedef util::Vector3D<LatticeSpeed> LatticeVelocity;
										//std::vector<double> wallMom_correction_received;
										double wallMom_correction_received;

										for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
										{
											geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
											kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

											for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
											{
												if (site.HasIolet(direction))
												{
													ioletLinkDelegate.Eval_wallMom_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_correction_received);
													// printf("Entering Branch StreamerTypeFactory 2 \n"); // Yes... Enters this path
													//if (siteIndex==10065 ){ // && wallMom_correction_received!=0// && bValues->GetTimeStep()==1000)
													//	//printf("\n/LaddIolet/ Time: %d - Site Id: %ld, Dir: %d, correction: %.5e \n", bValues->GetTimeStep(), site.GetIndex(), direction, wallMom_correction_received);
													//	printf("\n/LaddIolet/ Site Id: %ld, Dir: %d, correction: %.5e \n", site.GetIndex(), direction, wallMom_correction_received);
													//}
												}
												else
												{
													wallMom_correction_received = 0.0;
												}

												result_wallMom_correction_Direct.push_back(wallMom_correction_received);
											}
										}

										wallMom_correction_Iolet = result_wallMom_correction_Direct;

									} // End of inline void DoGetWallMom_correction_Direct


								//
								// Get the prefactor associated with the correction term associated with the Wall momemtum term (single term value) - to be passed to the GPU
								// Case of (B) Iolets with Walls collision-streaming type - Vel BCs (LADDIOLET)
								// TODO: Actually hydrovars is not used by Eval_wallMom_correction - can be removed from the arguments' list
								template<bool tDoRayTracing>
										inline void DoGetWallMom_prefactor_correction_Direct(const site_t firstIndex,
																const site_t siteCount,
																const LbmParameters* lbmParams,
																geometry::LatticeData* latDat,
																lb::MacroscopicPropertyCache& propertyCache,
																std::vector<double>& wallMom_prefactor_correction_Iolet)
										{

											std::vector<distribn_t> result_wallMom_prefactor_correction_Direct;
											result_wallMom_prefactor_correction_Direct.reserve(siteCount*LatticeType::NUMVECTORS);

											double wallMom_prefactor_correction_received;

											for (site_t siteIndex = firstIndex; siteIndex < (firstIndex + siteCount); siteIndex++)
											{
												geometry::Site<geometry::LatticeData> site = latDat->GetSite(siteIndex);
												kernels::HydroVars<typename CollisionType::CKernel> hydroVars(site);

												for (Direction direction = 0; direction < LatticeType::NUMVECTORS; direction++)
												{
													if (site.HasIolet(direction))
													{
														ioletLinkDelegate.Eval_wallMom_prefactor_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_prefactor_correction_received);

														/*
														// Debugging
														if(siteIndex==28433 || siteIndex ==28612){
															if(wallMom_prefactor_correction_received!=0 ){
																printf("Entering Branch StreamerTypeFactory 2 \n"); // Yes... Enters this path
																printf("/StreamerTypeFactory - LaddIolet/ - Site Id: %ld, Dir: %d, prefactor_correction received: %.5e \n\n", site.GetIndex(), direction, wallMom_prefactor_correction_received);
															}
														}
														*/
														//
													}
													else
													{
														wallMom_prefactor_correction_received = 0.0;
													}

													result_wallMom_prefactor_correction_Direct.push_back(wallMom_prefactor_correction_received);
												}
											}


											wallMom_prefactor_correction_Iolet = result_wallMom_prefactor_correction_Direct;

										} // End of inline void DoGetWallMom_prefactor_correction_Direct
#endif
			};
		}
	}
}
#endif // HEMELB_LB_STREAMERS_STREAMERTYPEFACTORY_H
