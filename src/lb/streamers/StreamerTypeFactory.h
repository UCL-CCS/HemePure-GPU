
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
							std::vector<double> wallMom_Vect_x;
							std::vector<double> wallMom_Vect_y;
							std::vector<double> wallMom_Vect_z;

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
							}
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
							std::vector<double> wallMom_Vect_x;
							std::vector<double> wallMom_Vect_y;
							std::vector<double> wallMom_Vect_z;

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
						}
#endif
			};
		}
	}
}
#endif // HEMELB_LB_STREAMERS_STREAMERTYPEFACTORY_H
