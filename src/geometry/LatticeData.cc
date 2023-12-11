
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include <map>
#include <limits>

#include "log/Logger.h"
#include "net/IOCommunicator.h"
#include "geometry/BlockTraverser.h"
#include "geometry/LatticeData.h"
#include "geometry/neighbouring/NeighbouringLatticeData.h"
#include "util/utilityFunctions.h"

#include <omp.h>
#include <algorithm>

#ifdef HEMELB_USE_GPU
/* Set the pointers to null so we don't do any accidental freeing at the end */
#define GPU_INITIALIZERS     , GPUDataAddr_dbl_fOld_b_mLatDat(nullptr),\
							 GPUDataAddr_dbl_fNew_b_mLatDat(nullptr),\
							 d_Stability_GPU_mLatDat(nullptr), \
							 GPUDataAddr_Inlet_velocityTable(nullptr),\
							 GPUDataAddr_Outlet_velocityTable(nullptr)
		
#else

/* No GPU use */
#define GPU_INITIALIZERS 

#endif

namespace hemelb
{
	namespace geometry
	{
		LatticeData::LatticeData(const lb::lattices::LatticeInfo& latticeInfo, const net::IOCommunicator& comms_) :
			latticeInfo(latticeInfo), neighbouringData(new neighbouring::NeighbouringLatticeData(latticeInfo)), comms(comms_) GPU_INITIALIZERS
		{
		}

		LatticeData::~LatticeData()
		{
			delete neighbouringData;
		}

		LatticeData::LatticeData(const lb::lattices::LatticeInfo& latticeInfo, const Geometry& readResult, const net::IOCommunicator& comms_) :
			latticeInfo(latticeInfo), neighbouringData(new neighbouring::NeighbouringLatticeData(latticeInfo)), comms(comms_) GPU_INITIALIZERS
		{

			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> LatticeDataInitializer: sorting Nonempty blocks");
			nonEmptyBlocks.reserve(readResult.Blocks.size());
		    for( auto kv : readResult.Blocks )  nonEmptyBlocks.push_back(kv.first);
			std::sort(nonEmptyBlocks.begin(), nonEmptyBlocks.end());
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> LatticeDataInitializer: setting basic Details");
			SetBasicDetails(readResult.GetBlockDimensions(),
					readResult.GetBlockSize());

			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> LatticeDataInitializer: processing read sites");
			ProcessReadSites(readResult);

			// If debugging then output beliefs regarding geometry and neighbour list.
			if (log::Logger::ShouldDisplay<log::Trace>())
			{
				proc_t localRank = comms.Rank();
				for (std::vector<NeighbouringProcessor>::iterator itNeighProc = neighbouringProcs.begin();
						itNeighProc != neighbouringProcs.end(); ++itNeighProc)
				{
					log::Logger::Log<log::Trace, log::OnePerCore>("LatticeData: Rank %i thinks that rank %i is a neighbour with %i shared edges\n",
							localRank,
							itNeighProc->Rank,
							itNeighProc->SharedDistributionCount);
				}
			}
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> LatticeDataInitializer: Collect Fluid Distributions");

			CollectFluidSiteDistribution();
			CollectGlobalSiteExtrema(); 
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> LatticeDataInitializer: Initialize Neighbor lookups");

			InitialiseNeighbourLookups();
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> LatticeDataInitializer: done");

		}

		void LatticeData::SetBasicDetails(util::Vector3D<site_t> blocksIn,
				site_t blockSizeIn)
		{
			blockCounts = blocksIn;
			blockSize = blockSizeIn;
			sites = blocksIn * blockSize;
			sitesPerBlockVolumeUnit = blockSize * blockSize * blockSize;
			blockCount = blockCounts.x * blockCounts.y * blockCounts.z;
		}

		void LatticeData::ProcessReadSites(const Geometry & readResult)
		{
			double stime=omp_get_wtime();

			// This represents an empty block.
			// Necessary for map object.
			blocks[-1] = Block(0);

			totalSharedFs = 0;


			// Global ones
			std::vector<SiteData> domainEdgeSiteData[COLLISION_TYPES];
			std::vector<SiteData> midDomainSiteData[COLLISION_TYPES];
			std::vector<site_t> domainEdgeBlockNumber[COLLISION_TYPES];
			std::vector<site_t> midDomainBlockNumber[COLLISION_TYPES];
			std::vector<site_t> domainEdgeSiteNumber[COLLISION_TYPES];
			std::vector<site_t> midDomainSiteNumber[COLLISION_TYPES];
			std::vector<util::Vector3D<float> > domainEdgeWallNormals[COLLISION_TYPES];
			std::vector<util::Vector3D<float> > midDomainWallNormals[COLLISION_TYPES];
			std::vector<float> domainEdgeWallDistance[COLLISION_TYPES];
			std::vector<float> midDomainWallDistance[COLLISION_TYPES];

			// Per thread onea
{

			const size_t MT = omp_get_max_threads();
			std::vector<SiteData> domainEdgeSiteDataT[COLLISION_TYPES][ MT ];
			std::vector<SiteData> midDomainSiteDataT[COLLISION_TYPES][ MT ];
			std::vector<site_t> domainEdgeBlockNumberT[COLLISION_TYPES][ MT ];
			std::vector<site_t> midDomainBlockNumberT[COLLISION_TYPES][ MT ];
			std::vector<site_t> domainEdgeSiteNumberT[COLLISION_TYPES][ MT ];
			std::vector<site_t> midDomainSiteNumberT[COLLISION_TYPES][ MT ] ;
			std::vector<util::Vector3D<float> > domainEdgeWallNormalsT[COLLISION_TYPES][ MT ];
			std::vector<util::Vector3D<float> > midDomainWallNormalsT[COLLISION_TYPES][ MT ];
			std::vector<float> domainEdgeWallDistanceT[COLLISION_TYPES][ MT ];
			std::vector<float> midDomainWallDistanceT[COLLISION_TYPES][ MT ];
			std::vector<LatticeForceVector> forceAtSiteT[MT];

			std::map< size_t , std::pair< size_t, size_t > > neighbourRankMap;

			proc_t localRank = comms.Rank();
			util::Vector3D<site_t> GeomDims = (*this).GetBlockDimensions();
			const site_t numBlocks = GeomDims.x * GeomDims.y * GeomDims.z;
			const site_t numNonEmptyBlocks = nonEmptyBlocks.size();
			const site_t numSites = blockSize*blockSize*blockSize;
	
			#pragma omp parallel for
            for( uint64_t bb=0; bb < numNonEmptyBlocks; bb++) {
                site_t blockId = nonEmptyBlocks[bb];
				#pragma omp critical
				{
                    blocks[blockId] = Block(GetSitesPerBlockVolumeUnit());
                }
            }

			
			#pragma omp parallel for 
			for( uint64_t bs=0; bs < numNonEmptyBlocks*numSites; bs++) { 
				uint64_t bb = bs / numSites;
				uint64_t localSiteId = bs % numSites;
				site_t blockId = nonEmptyBlocks[bb];
				const BlockReadResult & blockReadIn = readResult.Blocks.at(blockId);
				blocks.at(blockId).SetProcessorRankForSite(localSiteId, blockReadIn.Sites[localSiteId].targetProcessor);
			}

		
			#pragma omp parallel for reduction(+:totalSharedFs)
			for( int bb=0; bb < nonEmptyBlocks.size(); bb++) {
				site_t blockId = nonEmptyBlocks[bb];
				int tid = omp_get_thread_num();

				const BlockReadResult & blockReadIn = readResult.Blocks.at(blockId);
				if( blockReadIn.principalProcForBlock != localRank ) {
					continue;
				}

				util::Vector3D<site_t> currentBlockLocation = GetBlockCoordsFromBlockID( blockId );

				for( site_t localSiteId = 0; localSiteId < numSites; localSiteId++ ) 
				{
					// If the site is a solid continue 
					if (localRank != blockReadIn.Sites[localSiteId].targetProcessor)
					{
						continue;
					}
					bool isMidDomainSite = true;

					// Iterate over all non-zero direction vectors.
					for (unsigned int l = 1; l < latticeInfo.GetNumVectors(); l++)
					{
						// Find the neighbour site co-ords in this direction.
						util::Vector3D<site_t> neighbourGlobalCoords = currentBlockLocation
							* blockSize + GetSiteCoordsFromSiteID(localSiteId)
							+ util::Vector3D<site_t>(latticeInfo.GetVector(l));

						// If the neighbor is outside the Geometry domain skip.
						if (neighbourGlobalCoords.x < 0 || neighbourGlobalCoords.y < 0 || neighbourGlobalCoords.z < 0
								|| neighbourGlobalCoords.x >= readResult.GetBlockDimensions().x * blockSize 
								|| neighbourGlobalCoords.y >= readResult.GetBlockDimensions().y * blockSize
								|| neighbourGlobalCoords.z >= readResult.GetBlockDimensions().z * blockSize)
						{
							continue;
						}

						// Reduce the Global Neighbor coordinate to a Block coordinate and a site coordinate
						util::Vector3D<site_t> neighbourBlock = neighbourGlobalCoords / blockSize;
						util::Vector3D<site_t> neighbourSite  = neighbourGlobalCoords % blockSize;

					    // Get the neighbor block ID 
						site_t neighbourBlockId = readResult.GetBlockIdFromBlockCoordinates(neighbourBlock.x,
								neighbourBlock.y,
								neighbourBlock.z);

						// Move on if the neighbour is in a block of solids
						// in which case the block will contain zero sites
						// Or on if the neighbour site is solid
						// in which case the targetProcessor is SITE_OR_BLOCK_SOLID
						// Or the neighbour is also on this processor
						// in which case the targetProcessor is localRank
						if (readResult.Blocks.find(neighbourBlockId) == readResult.Blocks.end())
						{
							continue;
						}

						site_t neighbourSiteId = readResult.GetSiteIdFromSiteCoordinates(
								neighbourSite.x,
								neighbourSite.y,
								neighbourSite.z);

						proc_t neighbourProc = readResult.Blocks.at(neighbourBlockId).Sites[neighbourSiteId].targetProcessor;
						if (neighbourProc == SITE_OR_BLOCK_SOLID || localRank == neighbourProc)
						{
							continue;
						}

						// This is a local value
						isMidDomainSite = false;

						// This is shared and the loop here is essentially a reduction over this
						totalSharedFs+=1;

						// direction blocksite index for sorting
						size_t map_idx = l + latticeInfo.GetNumVectors()*( localSiteId + numSites*bb );

						#pragma omp critical
						{
						auto it = neighbourRankMap.find( neighbourProc );
			
						if( it != neighbourRankMap.end() ) {
							(*it).second.first++;
						}
						else {
							// Won't make duplicate entries but then if it is a duplicate we should find it
							neighbourRankMap[ neighbourProc ] = std::make_pair(1, map_idx);
						}
						}; // critical

#if 0	
						// The first time, net_neigh_procs = 0, so
						// the loop is not executed.
						bool flag = true;
						// Iterate over neighbouring processors until we find the one with the
						// neighbouring site on it.
						proc_t lNeighbouringProcs = (proc_t) ( (neighbouringProcs.size()));
						for (proc_t mm = 0; mm < lNeighbouringProcs && flag; mm++)
						{
							// Check whether the rank for a particular neighbour has already been
							// used for this processor.  If it has, set flag to zero.
							NeighbouringProcessor* neigh_proc_p = &neighbouringProcs[mm];
							// If ProcessorRankForEachBlockSite is equal to a neigh_proc that has alredy been listed.
							if (neighbourProc == neigh_proc_p->Rank)
							{
								flag = false;
								neigh_proc_p->SharedDistributionCount += 1;
								break;
							}
						}

						// If flag is 1, we didn't find a neighbour-proc with the neighbour-site on it
						// so we need a new neighbouring processor. We push back on a shared list
						// If every thread kept its own list of neighboring processors we could
						// avoid the lock. and the atomic above. But then we would need to merge 
						// the lists at the end.
						if (flag)
						{
							// Store rank of neighbour in >neigh_proc[neigh_procs].
							NeighbouringProcessor lNewNeighbour;
							lNewNeighbour.SharedDistributionCount = 1;
							lNewNeighbour.Rank = neighbourProc;

							neighbouringProcs.push_back(lNewNeighbour);

							// If debugging then output decisions with reasoning for all neighbour processors.
							log::Logger::Log<log::Trace, log::OnePerCore>("LatticeData: added %i as neighbour for %i because site %i in block %i is neighbour to site %i in block %i in direction (%i,%i,%i)\n",
									(int) neighbourProc,
									(int) localRank,
									(int) neighbourSiteId,
									(int) neighbourBlockId,
									(int) localSiteId,
									(int) blockId,
									latticeInfo.GetVector(l).x,
									latticeInfo.GetVector(l).y,
									latticeInfo.GetVector(l).z);
						}
    #endif
					} // directions

					// Set the collision type data. map_block site data is renumbered according to
					// fluid site numbers within a particular collision type.
					SiteData siteData(blockReadIn.Sites[localSiteId]);
					int l = -1;
					switch (siteData.GetCollisionType())
					{
						case FLUID:
							l = 0;
							break;
						case WALL:
							l = 1;
							break;
						case INLET:
							l = 2;
							break;
						case OUTLET:
							l = 3;
							break;
						case (INLET | WALL):
							l = 4;
							break;
						case (OUTLET | WALL):
							l = 5;
							break;
					}

					const util::Vector3D<float>& normal = blockReadIn.Sites[localSiteId].wallNormalAvailable ?
						blockReadIn.Sites[localSiteId].wallNormal :
						util::Vector3D<float>(NO_VALUE);

					forceAtSiteT[tid].push_back(LatticeForceVector(0, 0, 0));

					if ( isMidDomainSite ) 
					{
						midDomainBlockNumberT[l][tid].push_back(blockId);
						midDomainSiteNumberT[l][tid].push_back(localSiteId);
						midDomainSiteDataT[l][tid].push_back(siteData);
						midDomainWallNormalsT[l][tid].push_back(normal);
						for (Direction direction = 1; direction < latticeInfo.GetNumVectors(); direction++)
						{
							midDomainWallDistanceT[l][tid].push_back(blockReadIn.Sites[localSiteId].links[direction - 1].distanceToIntersection);
						}
					}
					else
					{
						domainEdgeBlockNumberT[l][tid].push_back(blockId);
						domainEdgeSiteNumberT[l][tid].push_back(localSiteId);
						domainEdgeSiteDataT[l][tid].push_back(siteData);
						domainEdgeWallNormalsT[l][tid].push_back(normal);
						for (Direction direction = 1; direction < latticeInfo.GetNumVectors(); direction++)
						{
							domainEdgeWallDistanceT[l][tid].push_back(blockReadIn.Sites[localSiteId].links[direction - 1].distanceToIntersection);
						}
					}
				} // sites
			} // blocks

			for(int tid =0; tid < MT; ++tid) {
				forceAtSite.insert( forceAtSite.end(), forceAtSiteT[tid].begin(), forceAtSiteT[tid].end());
			}

			for(int l=0; l < COLLISION_TYPES; l++)  {
				for(int tid=0; tid < MT; ++tid) { 
					midDomainBlockNumber[l].insert( midDomainBlockNumber[l].end(), midDomainBlockNumberT[l][tid].begin(), midDomainBlockNumberT[l][tid].end() );
					midDomainSiteNumber[l].insert( midDomainSiteNumber[l].end(), midDomainSiteNumberT[l][tid].begin(), midDomainSiteNumberT[l][tid].end() );
					midDomainSiteData[l].insert( midDomainSiteData[l].end(),  midDomainSiteDataT[l][tid].begin(),  midDomainSiteDataT[l][tid].end() );
					midDomainWallNormals[l].insert( midDomainWallNormals[l].end(), midDomainWallNormalsT[l][tid].begin(), midDomainWallNormalsT[l][tid].end());
					midDomainWallDistance[l].insert( midDomainWallDistance[l].end(), midDomainWallDistanceT[l][tid].begin(), midDomainWallDistanceT[l][tid].end());
 

					domainEdgeBlockNumber[l].insert( domainEdgeBlockNumber[l].end(), domainEdgeBlockNumberT[l][tid].begin(), domainEdgeBlockNumberT[l][tid].end() );
					domainEdgeSiteNumber[l].insert( domainEdgeSiteNumber[l].end(), domainEdgeSiteNumberT[l][tid].begin(), domainEdgeSiteNumberT[l][tid].end() );
					domainEdgeSiteData[l].insert( domainEdgeSiteData[l].end(),  domainEdgeSiteDataT[l][tid].begin(),  domainEdgeSiteDataT[l][tid].end() );
					domainEdgeWallNormals[l].insert( domainEdgeWallNormals[l].end(), domainEdgeWallNormalsT[l][tid].begin(), domainEdgeWallNormalsT[l][tid].end());
					domainEdgeWallDistance[l].insert( domainEdgeWallDistance[l].end(), domainEdgeWallDistanceT[l][tid].begin(), domainEdgeWallDistanceT[l][tid].end());
				}
			}

			// Sort out the neighbor procs
			std::map<site_t, NeighbouringProcessor  > tmpmap;
			for( auto kv : neighbourRankMap ) {
				proc_t rank = static_cast<proc_t>(kv.first);
				site_t shared_fs = static_cast<site_t>(kv.second.first);
				size_t bs = kv.second.second;
				NeighbouringProcessor np; 
				np.Rank = rank; np.SharedDistributionCount = shared_fs;
				tmpmap[bs] = np ;
			}
			for( auto kv : tmpmap) {
				neighbouringProcs.push_back( kv.second );
			}

} // threaded things go out of scpoe
			
			PopulateWithReadData(midDomainBlockNumber,
					midDomainSiteNumber,
					midDomainSiteData,
					midDomainWallNormals,
					midDomainWallDistance,
					domainEdgeBlockNumber,
					domainEdgeSiteNumber,
					domainEdgeSiteData,
					domainEdgeWallNormals,
					domainEdgeWallDistance);
		}

		void LatticeData::CollectFluidSiteDistribution()
		{
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> gathering lattice information (start)");
			fluidSitesOnEachProcessor = comms.AllGather(localFluidSites);
			totalFluidSites = 0;

#pragma omp parallel for reduction(+:totalFluidSites)
			for (proc_t ii = 0; ii < comms.Size(); ++ii)
			{
				totalFluidSites += fluidSitesOnEachProcessor[ii];
			}
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> gathering lattice information (end)");
		}


		// In reality this is a reduction ( mins over 3D and maxes over 3D)
		void LatticeData::CollectGlobalSiteExtrema()
		{
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> collecting site extrema (start)");
			site_t localMin0 = std::numeric_limits<site_t>::max();
			site_t localMin1 = std::numeric_limits<site_t>::max();
			site_t localMin2 = std::numeric_limits<site_t>::max();
			site_t localMax0 = 0;
			site_t localMax1 = 0;
			site_t localMax2 = 0;

			//for (geometry::BlockTraverser blockSet(*this); blockSet.CurrentLocationValid(); blockSet.TraverseOne())
			//{
				// const geometry::Block& block = blockSet.GetCurrentBlockData();
			const site_t numSites = blockSize*blockSize*blockSize;

			#pragma omp parallel for reduction( max : localMax0, localMax1, localMax2) reduction( min : localMin0, localMin1, localMin2 ) collapse(2)
			for( size_t bb = 0; bb < nonEmptyBlocks.size(); bb++ ) { 
				for(site_t siteId=0; siteId < numSites; siteId++)  {
					site_t blockId = nonEmptyBlocks[bb];
					util::Vector3D<site_t> currentBlockCoords = GetBlockCoordsFromBlockID( blockId );
				    const geometry::Block& block = blocks.at(blockId);
					if (block.GetProcessorRankForSite(siteId) == comms.Rank()) {

						util::Vector3D<site_t> globalCoords = currentBlockCoords * GetBlockSize()
							+ GetSiteCoordsFromSiteID(siteId);

						localMin0 = hemelb::util::NumericalFunctions::min(localMin0, globalCoords[0]);
						localMin1 = hemelb::util::NumericalFunctions::min(localMin1, globalCoords[1]);
						localMin2 = hemelb::util::NumericalFunctions::min(localMin2, globalCoords[2]);
						localMax0 = hemelb::util::NumericalFunctions::max(localMax0, globalCoords[0]);
						localMax1 = hemelb::util::NumericalFunctions::max(localMax1, globalCoords[1]);
						localMax2 = hemelb::util::NumericalFunctions::max(localMax2, globalCoords[2]);
					}
				}
			}
			std::vector<site_t> localMins{ localMin0, localMin1, localMin2 };
			std::vector<site_t> localMaxes{ localMax0, localMax1, localMax2 };

			std::vector<site_t> siteMins = comms.AllReduce(localMins, MPI_MIN);
			std::vector<site_t> siteMaxes = comms.AllReduce(localMaxes, MPI_MAX);

			for (unsigned ii = 0; ii < 3; ++ii)
			{
				globalSiteMins[ii] = siteMins[ii];
				globalSiteMaxes[ii] = siteMaxes[ii];
			}
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> collecting site extrema (end)");
		}

		void LatticeData::InitialiseNeighbourLookups()
		{
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> initializing neighbour lookups (start)");
			// Allocate the index in which to put the distribution functions received from the other
			// process.
			std::vector< std::vector<site_t> > sharedDistributionLocationForEachProc(comms.Size());

			site_t totalSharedDistributionsSoFar = 0;
			// Set the remaining neighbouring processor data.
			for (size_t neighbourId = 0; neighbourId < neighbouringProcs.size(); neighbourId++)
			{
				// Pointing to a few things, but not setting any variables.
				// FirstSharedF points to start of shared_fs.
				neighbouringProcs[neighbourId].FirstSharedDistribution = GetLocalFluidSiteCount()
					* latticeInfo.GetNumVectors() + 1 + totalSharedDistributionsSoFar;
				totalSharedDistributionsSoFar += neighbouringProcs[neighbourId].SharedDistributionCount;
			}
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> initializing neighbour lookups: NLookup");
			InitialiseNeighbourLookup(sharedDistributionLocationForEachProc);
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> initializing neighbour lookups: Point to Point comms");
			InitialisePointToPointComms(sharedDistributionLocationForEachProc);
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> initializing neighbour lookups: Receive Lookup");
			InitialiseReceiveLookup(sharedDistributionLocationForEachProc);
			hemelb::log::Logger::Log<hemelb::log::Info, hemelb::log::Singleton>("----> initializing neighbour lookups (end)");
		}

		void LatticeData::InitialiseNeighbourLookup(std::vector<std::vector<site_t> >& sharedFLocationForEachProc)
		{
			const proc_t localRank = comms.Rank();
			neighbourIndices.resize(latticeInfo.GetNumVectors() * localFluidSites);

			// These will be private to each thread
			std::vector< std::vector< std::vector< site_t > > > threadLocalSharedDistributionLocations(omp_get_max_threads());
			#pragma omp parallel 
			{
				auto tid = omp_get_thread_num();
				threadLocalSharedDistributionLocations[ tid ].resize(comms.Size());
			}
#if 0
			for (BlockTraverser blockTraverser(*this); blockTraverser.CurrentLocationValid(); blockTraverser.TraverseOne())
			{
				const Block& map_block_p = blockTraverser.GetCurrentBlockData();
				if (map_block_p.IsEmpty())
				{
					continue;
				}
#endif

			// This should split the loop so that each thread is responsible for  
			// a contiguous number of blocks. So at the end we can concatenate the results.
			#pragma omp parallel for 
			for( site_t bb = 0; bb < nonEmptyBlocks.size(); bb++) {
				int tid = omp_get_thread_num();

				site_t blockId = nonEmptyBlocks[bb];
				const Block& map_block_p = blocks.at( blockId );
#if 0
				for (SiteTraverser siteTraverser = blockTraverser.GetSiteTraverser();
						siteTraverser.CurrentLocationValid(); siteTraverser.TraverseOne())
#endif
				const util::Vector3D<site_t> currBlockCoords = GetBlockCoordsFromBlockID( blockId );

				const site_t numSites = blockSize* blockSize * blockSize;

				for( site_t siteID = 0 ; siteID  < numSites; siteID++)  {
					const util::Vector3D<site_t> currSite = GetSiteCoordsFromSiteID(siteID);

					if (localRank != map_block_p.GetProcessorRankForSite(siteID))
					{
						continue;
					}
					// Get site data, which is the number of the fluid sites on this proc.
					site_t localIndex = map_block_p.GetLocalContiguousIndexForSite(siteID);

					// Set neighbour location for the distribution component at the centre of
					// this site.
					SetNeighbourLocation(localIndex, 0, localIndex * latticeInfo.GetNumVectors() + 0);
					for (Direction direction = 1; direction < latticeInfo.GetNumVectors(); direction++)
					{
						util::Vector3D<site_t> currentLocationCoords = currBlockCoords * blockSize +  currSite;
						// Work out positions of neighbours.
						util::Vector3D<site_t> neighbourCoords = currentLocationCoords + util::Vector3D<site_t>(latticeInfo.GetVector(direction));
						if (!IsValidLatticeSite(neighbourCoords))
						{
							// Set the neighbour location to the rubbish site.
							SetNeighbourLocation(localIndex, direction, GetLocalFluidSiteCount() * latticeInfo.GetNumVectors());
							continue;
						}
						// Get the id of the processor which the neighbouring site lies on.
						const proc_t proc_id_p = GetProcIdFromGlobalCoords(neighbourCoords);
						if (proc_id_p == SITE_OR_BLOCK_SOLID)
						{
							// initialize f_id to the rubbish site.
							SetNeighbourLocation(localIndex, direction, GetLocalFluidSiteCount() * latticeInfo.GetNumVectors());
							continue;
						}
						else {
							// If on the same proc, set f_id of the
							// current site and direction to the
							// site and direction that it sends to.
							// If we check convergence, the data for
							// each site is split into that for the
							// current and previous cycles.
							if (localRank == proc_id_p)
							{
								// Pointer to the neighbour.
								site_t contigSiteId = GetContiguousSiteId(neighbourCoords);
								SetNeighbourLocation(localIndex, direction, contigSiteId * latticeInfo.GetNumVectors() + direction);
								continue;
							}
							else
							{
								// This stores some coordinates.  We
								// still need to know the site number.
								// neigh_proc[ n ].f_data is now
								// set as well, since this points to
								// f_data.  Every process has data for
								// its neighbours which say which sites
								// on this process are shared with the
								// neighbour.

								threadLocalSharedDistributionLocations[ tid ][ proc_id_p ].push_back( currentLocationCoords.x);
								threadLocalSharedDistributionLocations[ tid ][ proc_id_p ].push_back( currentLocationCoords.y);
								threadLocalSharedDistributionLocations[ tid ][ proc_id_p ].push_back( currentLocationCoords.z);
								threadLocalSharedDistributionLocations[ tid ][ proc_id_p ].push_back( direction );
							}
						}
					}
				}
			}// loop over blocks

			// now concatenate the blocks
			// We can parallelize this over the processors
			#pragma omp parallel for 
			for(unsigned int proc = 0; proc < comms.Size(); proc++) {
				sharedFLocationForEachProc[proc].clear();
				for(unsigned int tid=0; tid < omp_get_max_threads(); tid++) { 
				  sharedFLocationForEachProc[proc].insert( sharedFLocationForEachProc[proc].end(), threadLocalSharedDistributionLocations[ tid ][ proc ].begin(), threadLocalSharedDistributionLocations[ tid ][ proc ].end() );
				}
			}
	
			
		}// function

		void LatticeData::InitialisePointToPointComms(std::vector<std::vector<site_t> >& sharedFLocationForEachProc)
		{
			proc_t localRank = comms.Rank();
			// point-to-point communications are performed to match data to be
			// sent to/receive from different partitions; in this way, the
			// communication of the locations of the interface-dependent fluid
			// sites and the identifiers of the distribution functions which
			// propagate to different partitions is avoided (only their values
			// will be communicated). It's here!
			// Allocate the request variable.
			net::Net tempNet(comms);
			for (size_t neighbourId = 0; neighbourId < neighbouringProcs.size(); neighbourId++)
			{
				NeighbouringProcessor* neigh_proc_p = &neighbouringProcs[neighbourId];
				// One way send receive.  The lower numbered netTop->ProcessorCount send and the higher numbered ones receive.
				// It seems that, for each pair of processors, the lower numbered one ends up with its own
				// edge sites and directions stored and the higher numbered one ends up with those on the
				// other processor.
				if (neigh_proc_p->Rank > localRank)
				{
					tempNet.RequestSendV(sharedFLocationForEachProc[neigh_proc_p->Rank], neigh_proc_p->Rank);
				}
				else
				{
					sharedFLocationForEachProc[neigh_proc_p->Rank].resize(neigh_proc_p->SharedDistributionCount * 4);
					tempNet.RequestReceiveV(sharedFLocationForEachProc[neigh_proc_p->Rank], neigh_proc_p->Rank);
				}
			}
			tempNet.Dispatch();
		}

		void LatticeData::InitialiseReceiveLookup(std::vector<std::vector<site_t> >& sharedFLocationForEachProc)
		{
			proc_t localRank = comms.Rank();
			streamingIndicesForReceivedDistributions.resize(totalSharedFs);
			site_t f_count = GetLocalFluidSiteCount() * latticeInfo.GetNumVectors();
			site_t sharedSitesSeen = 0;

			for (size_t neighbourId = 0; neighbourId < neighbouringProcs.size(); neighbourId++)
			{
				NeighbouringProcessor* neigh_proc_p = &neighbouringProcs[neighbourId];
				for (site_t sharedDistributionId = 0; sharedDistributionId < neigh_proc_p->SharedDistributionCount;
						sharedDistributionId++)
				{
					// Get coordinates and direction of the distribution function to be sent to another process.
					site_t* f_data_p = &sharedFLocationForEachProc[neigh_proc_p->Rank][sharedDistributionId * 4];
					site_t i = f_data_p[0];
					site_t j = f_data_p[1];
					site_t k = f_data_p[2];
					site_t l = f_data_p[3];
					// Correct so that each process has the correct coordinates.
					if (neigh_proc_p->Rank < localRank)
					{
						i += latticeInfo.GetVector(l).x;
						j += latticeInfo.GetVector(l).y;
						k += latticeInfo.GetVector(l).z;
						l = latticeInfo.GetInverseIndex(l);
					}

					// Get the fluid site number of site that will send data to another process.
					util::Vector3D<site_t> location(i, j, k);
					site_t contigSiteId = GetContiguousSiteId(location);
					// Set f_id to the element in the send buffer that we put the updated
					// distribution functions in.
					SetNeighbourLocation(contigSiteId, (unsigned int) ( (l)), ++f_count);
					// Set the place where we put the received distribution functions, which is
					// f_new[number of fluid site that sends, inverse direction].
					streamingIndicesForReceivedDistributions[sharedSitesSeen] = contigSiteId * latticeInfo.GetNumVectors()
						+ latticeInfo.GetInverseIndex(l);
					++sharedSitesSeen;
				}
			}
		}

		proc_t LatticeData::GetProcIdFromGlobalCoords(const util::Vector3D<site_t>& globalSiteCoords) const
		{
			// Block identifiers (i, j, k) of the site (site_i, site_j, site_k).
			util::Vector3D<site_t> blockCoords, localSiteCoords;
			GetBlockAndLocalSiteCoords(globalSiteCoords, blockCoords, localSiteCoords);
			// Get the block from the block identifiers.
			const Block& block = GetBlock(GetBlockIdFromBlockCoords(blockCoords));
			// If an empty (solid) block is addressed, return a NULL pointer.
			if (block.IsEmpty())
			{
				return SITE_OR_BLOCK_SOLID;
			}
			else
			{
				// Return pointer to ProcessorRankForEachBlockSite[site] (the only member of
				// mProcessorsForEachBlock).
				return block.GetProcessorRankForSite(GetLocalSiteIdFromLocalSiteCoords(localSiteCoords));
			}
		}

		bool LatticeData::IsValidBlock(site_t i, site_t j, site_t k) const
		{
			if (i < 0 || i >= blockCounts.x)
				return false;
			if (j < 0 || j >= blockCounts.y)
				return false;
			if (k < 0 || k >= blockCounts.z)
				return false;

			return true;
		}

		bool LatticeData::IsValidBlock(const util::Vector3D<site_t>& blockCoords) const
		{
			if (blockCoords.x < 0 || blockCoords.x >= blockCounts.x)
				return false;
			if (blockCoords.y < 0 || blockCoords.y >= blockCounts.y)
				return false;
			if (blockCoords.z < 0 || blockCoords.z >= blockCounts.z)
				return false;

			return true;
		}

		bool LatticeData::IsValidLatticeSite(const util::Vector3D<site_t>& siteCoords) const
		{
			if (siteCoords.x < 0 || siteCoords.x >= sites.x)
				return false;
			if (siteCoords.y < 0 || siteCoords.y >= sites.y)
				return false;
			if (siteCoords.z < 0 || siteCoords.z >= sites.z)
				return false;

			return true;
		}

		site_t LatticeData::GetContiguousSiteId(util::Vector3D<site_t> location) const
		{
			// Block identifiers (i, j, k) of the site (site_i, site_j, site_k).
			util::Vector3D<site_t> blockCoords, localSiteCoords;
			GetBlockAndLocalSiteCoords(location, blockCoords, localSiteCoords);
			// Pointer to the block.
			const Block& lBlock = GetBlock(GetBlockIdFromBlockCoords(blockCoords));
			// Return pointer to site_data[site].
			return lBlock.GetLocalContiguousIndexForSite(GetLocalSiteIdFromLocalSiteCoords(localSiteCoords));
		}

		bool LatticeData::GetContiguousSiteId(const util::Vector3D<site_t>& globalLocation,
				proc_t& procId,
				site_t& siteId) const
		{
			// Convert global coordinates to local coordinates - i.e.
			// to location of block and location of site within block.
			util::Vector3D<site_t> blockCoords, localSiteCoords;
			GetBlockAndLocalSiteCoords(globalLocation, blockCoords, localSiteCoords);
			if (!IsValidBlock(blockCoords) || !IsValidLatticeSite(localSiteCoords))
				return false;

			// Get information for the block using the block location.
			const Block& block = GetBlock(GetBlockIdFromBlockCoords(blockCoords));
			if (block.IsEmpty())
				return false;

			// Get the local site id, i.e. its index within the block.
			site_t localSiteIndex = GetLocalSiteIdFromLocalSiteCoords(localSiteCoords);

			// Get the rank of the processor that owns the site.
			procId = block.GetProcessorRankForSite(localSiteIndex);
			if (procId != comms.Rank())
				return false;
			if (procId == SITE_OR_BLOCK_SOLID) // Means that the site is solid.
				return false;

			// We only know enough information to determine solid/fluid for local sites.
			// Get the local contiguous index of the fluid site.
			if (block.SiteIsSolid(localSiteIndex))
				return false;
			else
				siteId = block.GetLocalContiguousIndexForSite(localSiteIndex);
			return true;
		}

		const util::Vector3D<site_t> LatticeData::GetGlobalCoords(site_t blockNumber,
				const util::Vector3D<site_t>& localSiteCoords) const
		{
			util::Vector3D<site_t> blockCoords;
			GetBlockIJK(blockNumber, blockCoords);
			return GetGlobalCoords(blockCoords, localSiteCoords);
		}

		util::Vector3D<site_t> LatticeData::GetSiteCoordsFromSiteId(site_t siteId) const
		{
			util::Vector3D<site_t> siteCoords;
			siteCoords.z = siteId % blockSize;
			site_t siteIJData = siteId / blockSize;
			siteCoords.y = siteIJData % blockSize;
			siteCoords.x = siteIJData / blockSize;
			return siteCoords;
		}

		void LatticeData::GetBlockAndLocalSiteCoords(const util::Vector3D<site_t>& location,
				util::Vector3D<site_t>& blockCoords,
				util::Vector3D<site_t>& siteCoords) const
		{
			blockCoords = location / blockSize;
			siteCoords = location % blockSize;
		}

		site_t LatticeData::GetMidDomainSiteCount() const
		{
			site_t midDomainSiteCount = 0;
			for (unsigned collisionType = 0; collisionType < COLLISION_TYPES; collisionType++)
			{
				midDomainSiteCount += midDomainProcCollisions[collisionType];
			}
			return midDomainSiteCount;
		}

		void LatticeData::GetBlockIJK(site_t block, util::Vector3D<site_t>& blockCoords) const
		{
			blockCoords.z = block % blockCounts.z;
			site_t blockIJData = block / blockCounts.z;
			blockCoords.y = blockIJData % blockCounts.y;
			blockCoords.x = blockIJData / blockCounts.y;
		}

		void LatticeData::SendAndReceive(hemelb::net::Net* net)
		{
			for (std::vector<NeighbouringProcessor>::const_iterator it = neighbouringProcs.begin();
					it != neighbouringProcs.end(); ++it)
			{

#ifdef HEMELB_CUDA_AWARE_MPI
				/**
						If cuda-aware mpi is enabled:
						then pass pointers to GPU global memory directly to the MPI calls
				*/

				int myPiD = GetLocalRank(); // Local rank
				/*
				std::cout << "CUDA-aware mpi branch: Current rank: " << myPiD << "   - Requesting Receive from : " \
								<< (*it).Rank  << " in location : " << (int) ( ( (*it).FirstSharedDistribution)) \
								<< " Number of elements : " << (int) ( ( (*it).SharedDistributionCount)) \
								<< std::endl;
				*/

				// Request the receive into the appropriate bit of FOld (Pointer to GPU global memory)
				// Replace 	GetFOld( (*it).FirstSharedDistribution)
				// 	with 		&(((distribn_t*)GPUDataAddr_dbl_fOld_b_mLatDat)[(int) ( ( (*it).FirstSharedDistribution))])
				net->RequestReceive<distribn_t>( &(((distribn_t*)GPUDataAddr_dbl_fOld_b_mLatDat)[(int) ( ( (int) ( ( (*it).FirstSharedDistribution))))]),
						(int) ( ( (*it).SharedDistributionCount)),
						(*it).Rank);

				// Request the send from the right bit of FNew (Pointer to GPU global memory)
				// Replace 	GetFNew( (*it).FirstSharedDistribution)
				//	with		&(((distribn_t*)GPUDataAddr_dbl_fNew_b_mLatDat)[(int) ( ( (*it).FirstSharedDistribution))])
				net->RequestSend<distribn_t>( &(((distribn_t*)GPUDataAddr_dbl_fNew_b_mLatDat)[(int) ( ( (*it).FirstSharedDistribution))]),
						(int) ( ( (*it).SharedDistributionCount)),
						(*it).Rank);

#else
				// Request the receive into the appropriate bit of FOld.
				net->RequestReceive<distribn_t>(GetFOld( (*it).FirstSharedDistribution),
						(int) ( ( (*it).SharedDistributionCount)),
						(*it).Rank);

				int myPiD = GetLocalRank(); // Local rank
				//std::cout << "NO CUDA-aware mpi branch: Current rank: " << myPiD << "   - Requesting Receive from : " << (*it).Rank  << " in location : " << (int) ( ( (*it).FirstSharedDistribution)) << std::endl;

				// Request the send from the right bit of FNew.
				net->RequestSend<distribn_t>(GetFNew( (*it).FirstSharedDistribution),
						(int) ( ( (*it).SharedDistributionCount)),
						(*it).Rank);
#endif

			}
		}

		void LatticeData::CopyReceived()
		{
			// Copy the distribution functions received from the neighbouring
			// processors into the destination buffer "f_new".
			for (site_t i = 0; i < totalSharedFs; i++)
			{
				*GetFNew(streamingIndicesForReceivedDistributions[i]) = *GetFOld(neighbouringProcs[0].FirstSharedDistribution + i);
			}
		}

		void LatticeData::Report(ctemplate::TemplateDictionary& dictionary)
		{
			dictionary.SetIntValue("SITES", GetTotalFluidSites());
			dictionary.SetIntValue("BLOCKS", blockCount);
			dictionary.SetIntValue("SITESPERBLOCK", sitesPerBlockVolumeUnit);
			for (size_t n = 0; n < fluidSitesOnEachProcessor.size(); n++)
			{
				ctemplate::TemplateDictionary *proc = dictionary.AddSectionDictionary("PROCESSOR");
				proc->SetIntValue("RANK", n);
				proc->SetIntValue("SITES", fluidSitesOnEachProcessor[n]);
			}
		}
		neighbouring::NeighbouringLatticeData &LatticeData::GetNeighbouringData()
		{
			return *neighbouringData;
		}
		neighbouring::NeighbouringLatticeData const & LatticeData::GetNeighbouringData() const
		{
			return *neighbouringData;
		}

		int LatticeData::GetLocalRank() const
		{
			return comms.Rank();
		}
	}
}
