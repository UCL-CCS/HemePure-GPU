
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include <cmath>
#include <list>
#include <map>
#include <algorithm>
#include <functional>
#include <numeric>
#include <zlib.h>

#include <iostream>
#include <fstream>
#include "io/formats/geometry.h"
#include "io/writers/xdr/XdrMemReader.h"
#include "geometry/decomposition/BasicDecomposition.h"
#include "geometry/decomposition/OptimisedDecomposition.h"
#include "geometry/GeometrySGMYReader.h"
#include "lb/lattices/D3Q27.h"
#include "net/net.h"
#include "net/IOCommunicator.h"
#include "log/Logger.h"
#include "util/utilityFunctions.h"
#include "constants.h"
#include <omp.h>


namespace hemelb
{
	namespace geometry
	{

		GeometrySGMYReader::GeometrySGMYReader(const lb::lattices::LatticeInfo& latticeInfo,
				reporting::Timers &atimings, const net::IOCommunicator& ioComm) :
			latticeInfo(latticeInfo), hemeLbComms(ioComm), timings(atimings)
		{
			// This rank should participate in the domain decomposition if
			//  - we're not on core 0 (the only core that might ever not participate)
			//  - there's only one processor (so core 0 has to participate)

			// Create our own group, without the root node if we're not running with it.
			if (ioComm.Size() > 1)
			{
				participateInTopology = !ioComm.OnIORank();

				std::vector<int> lExclusions(1);
				lExclusions[0] = 0;
				net::MpiGroup computeGroup = hemeLbComms.Group().Exclude(lExclusions);
				// Create a communicator just for the domain decomposition.
				computeComms = ioComm.Create(computeGroup);
			}
			else
			{
				participateInTopology = true;
				computeComms = ioComm;
			}
		}

		GeometrySGMYReader::~GeometrySGMYReader()
		{
		}

		Geometry GeometrySGMYReader::LoadAndDecompose(const std::string& dataFilePath)
		{
			timings[hemelb::reporting::Timers::fileRead].Start();

			// Create hints about how we'll read the file. See Chapter 13, page 400 of the MPI 2.2 spec.
			MPI_Info fileInfo;
			HEMELB_MPI_CALL(MPI_Info_create, (&fileInfo));
			std::string accessStyle = "access_style";
			std::string accessStyleValue = "sequential";
			std::string buffering = "collective_buffering";
			std::string bufferingValue = "true";

			HEMELB_MPI_CALL(MPI_Info_set, (fileInfo,
						const_cast<char*> (accessStyle.c_str()),
						const_cast<char*> (accessStyleValue.c_str()))
					);
			HEMELB_MPI_CALL(MPI_Info_set, (fileInfo,
						const_cast<char*> (buffering.c_str()),
						const_cast<char*> (bufferingValue.c_str()))
					);

			// Open the file.
			file = net::MpiFile::Open(hemeLbComms, dataFilePath, MPI_MODE_RDONLY, fileInfo);

			log::Logger::Log<log::Info, log::Singleton>("----> opened data file %s", dataFilePath.c_str());
			// TODO: Why is there this fflush?
			fflush(NULL);

			// Set the view to the file.
			file.SetView(0, MPI_CHAR, MPI_CHAR, "native", fileInfo);

			// The processor assigned to each block.
			std::unordered_map<site_t, proc_t>* principalProcForEachBlock =
				new std::unordered_map<site_t, proc_t>();

#ifdef HEMELB_USE_GMYPLUS
			log::Logger::Log<log::Info, log::Singleton>("----> accepting *.gmy+");
#endif
#ifdef HEMELB_USE_MPI_CALL
			log::Logger::Log<log::Info, log::Singleton>("----> using standard MPI calls in ReadInBlock()");
#endif
			log::Logger::Log<log::Info, log::Singleton>("----> reading SGMY preamble");

			SGMY::SGMYPreambleInfo preambleInfo;
			Geometry geometry = ReadPreamble(preambleInfo);

			log::Logger::Log<log::Info, log::Singleton>("----> reading header (start)");
			ReadHeader(preambleInfo);
			log::Logger::Log<log::Info, log::Singleton>("----> reading header (end)");

			// Close the file - only the ranks participating in the topology need to read it again.
			file.Close();

#if 0
			sitedata_t siteCount = 0;
			for (site_t block = 0; block < geometry.GetBlockCount(); ++block)
			{
				if (fluidSitesOnEachBlock.find(block) != fluidSitesOnEachBlock.end())
					siteCount += fluidSitesOnEachBlock.at(block);
			}
#else
			sitedata_t siteCount = 0;
			for(auto block = fluidSitesOnEachBlock.begin(); block != fluidSitesOnEachBlock.end(); block++) { 
			   siteCount += block->second;
			}
#endif
			log::Logger::Log<log::Info, log::Singleton>(
					"----> non-empty blocks: %lu",
					nonEmptyBlocks);
			log::Logger::Log<log::Info, log::Singleton>(
					"          total blocks: %lu",
					geometry.GetBlockCount());
			log::Logger::Log<log::Info, log::Singleton>(
					"                 ratio: %f",
					nonEmptyBlocks/(double)geometry.GetBlockCount());
			log::Logger::Log<log::Info, log::Singleton>(
					"                 sites: %lu",
					siteCount);

			log::Logger::Log<log::Info, log::Singleton>(
					"----> blockInformation.size(): %lu",
					blockInformation.size());
			log::Logger::Log<log::Info, log::Singleton>(
					" fluidSitesOnEachBlock.size(): %lu",
					fluidSitesOnEachBlock.size());
			log::Logger::Log<log::Info, log::Singleton>(
					"          blockWeights.size(): %lu",
					blockWeights.size());

			log::Logger::Log<log::Info, log::Singleton>("----> is blockInformation.size() == nonEmptyBlocks? %s", blockInformation.size() == nonEmptyBlocks ? "yes" : "no");

			site_t *principalProcForEachBlock_1;
			site_t *principalProcForEachBlock_1_local;
			proc_t *principalProcForEachBlock_2;
			proc_t *principalProcForEachBlock_2_local;

			char nodename[MPI_MAX_PROCESSOR_NAME];
			int nodestringlen;
			MPI_Get_processor_name(nodename, &nodestringlen);

			MPI_Comm nodecomm;
			MPI_Comm_split_type(hemeLbComms, MPI_COMM_TYPE_SHARED, hemeLbComms.Rank(),
					MPI_INFO_NULL, &nodecomm);

			int nodesize, noderank;
			MPI_Comm_size(nodecomm, &nodesize);
			MPI_Comm_rank(nodecomm, &noderank);

			log::Logger::Log<log::Info, log::Singleton>("----> basic decomposition (start)");
			timings[hemelb::reporting::Timers::initialDecomposition].Start();
			if (participateInTopology)
			{
				// Get an initial base-level decomposition of the domain macro-blocks over processors.
				// This will later be improved upon by ParMETIS.
				decomposition::BasicDecomposition basicDecomposer(geometry,
						latticeInfo,
						computeComms,
						blockInformation,
						blockWeights);
				basicDecomposer.Decompose(
						*principalProcForEachBlock);
			}
			timings[hemelb::reporting::Timers::initialDecomposition].Stop();
			log::Logger::Log<log::Info, log::Singleton>("----> basic decomposition (end)");

			// The processor assigned to each block we know about.
			std::unordered_map<site_t, proc_t>* principalProcForEachBlockFiltered =
				new std::unordered_map<site_t, proc_t>();


			log::Logger::Log<log::Info, log::Singleton>("----> read blocks (start)");
			// Perform the initial read-in.
			if (participateInTopology)
			{
				
				// Reopen in the file just between the nodes in the topology decomposition. Read in blocks
				// local to this node.
				file = net::MpiFile::Open(computeComms, dataFilePath, MPI_MODE_RDONLY, fileInfo);

				ReadInBlocksWithHalo(geometry,
						*principalProcForEachBlock,
						*principalProcForEachBlockFiltered,
						computeComms.Rank(), preambleInfo);
			}
			timings[hemelb::reporting::Timers::fileRead].Stop();

			log::Logger::Log<log::Info, log::Singleton>("----> read blocks (end)");
		
			timings[hemelb::reporting::Timers::domainDecomposition].Start();
			// Having done an initial decomposition of the geometry, and read in the data, we optimise the
			// domain decomposition.
			if (participateInTopology)
			{
				OptimiseDomainDecomposition(geometry, *principalProcForEachBlock, *principalProcForEachBlockFiltered);

				for( auto kv : *principalProcForEachBlockFiltered ) {
					site_t blockID = kv.first;
					proc_t proc = kv.second;
					geometry.Blocks[blockID].principalProcForBlock = ConvertTopologyRankToGlobalRank(proc);	
				}

				file.Close();
			}
			// Finish up - close the file, set the timings, deallocate memory.
			HEMELB_MPI_CALL(MPI_Info_free, (&fileInfo));
			timings[hemelb::reporting::Timers::domainDecomposition].Stop();


			delete principalProcForEachBlock;
			delete principalProcForEachBlockFiltered;

			return geometry;
		}

		std::vector<char> GeometrySGMYReader::ReadOnAllTasks(sitedata_t nBytes)
		{
			std::vector<char> buffer(nBytes);
			const net::MpiCommunicator& comm = file.GetCommunicator();
#ifndef HEMELB_USE_MPI_PARALLEL_IO
			if (comm.Rank() == HEADER_READING_RANK)
			{
				file.Read(buffer);
			}
			comm.Broadcast(buffer, HEADER_READING_RANK);
#else
			file.Read(buffer);
#endif

			return buffer;
		}

		/**
		 * Read in the section at the beginning of the config file.
		 */
		Geometry GeometrySGMYReader::ReadPreamble(SGMY::SGMYPreambleInfo& preambleInfo)
		{
			const size_t preambleBytes = sizeof(SGMY::SGMYPreambleInfo);
			std::vector<char> preambleBuffer = ReadOnAllTasks(preambleBytes);
			preambleInfo = *(reinterpret_cast<SGMY::SGMYPreambleInfo *>(&preambleBuffer[0]));

			// Check the value of the HemeLB magic number.
			if ( preambleInfo.HemeLBMagic != io::formats::HemeLbMagicNumber)
			{
				throw Exception() << "This file does not start with the HemeLB magic number."
					<< " Expected: " << uint32_t(io::formats::HemeLbMagicNumber)
					<< " Actual: " << preambleInfo.HemeLBMagic;
			}

			// Check the value of the geometry file magic number.
			if ( preambleInfo.SgmyMagic != SGMY::SgmyMagicNumber)
			{
				throw Exception() << "This file does not have the geometry magic number."
					<< " Expected: " << uint32_t(SGMY::SgmyMagicNumber)
					<< " Actual: " << preambleInfo.SgmyMagic;
			}

			if ( preambleInfo.Version != SGMY::SgmyVersionNumber)
			{
				throw Exception() << "Version number incorrect."
					<< " Supported: " << uint32_t(SGMY::SgmyVersionNumber)
					<< " Input: " << preambleInfo.Version;
			}

			log::Logger::Log<log::Info, log::Singleton>("------> Non Empty Blocks = %lu",
				 preambleInfo.NonEmptyBlocks);

			nonEmptyBlocks = preambleInfo.NonEmptyBlocks;

			log::Logger::Log<log::Info, log::Singleton>("------> Blocks = (%u,%u,%u)",
				 preambleInfo.BlocksX, preambleInfo.BlocksY, preambleInfo.BlocksZ);
			log::Logger::Log<log::Info, log::Singleton>("------> BlockSize = %u", 
				 preambleInfo.BlockSize);
			log::Logger::Log<log::Info, log::Singleton>("------> Max Compressed Block Size = %lf KiB",
				 (double)preambleInfo.MaxCompressedBytes/(double)(1024));
			log::Logger::Log<log::Info, log::Singleton>("------> Max Uncompressed Block Size = %lf KiB",
				 (double)preambleInfo.MaxUncompressedBytes/(double)(1024));

			// Variables we'll read.
			// We use temporary vars here, as they must be the same size as the type in the file
			// regardless of the internal type used.
			return Geometry(util::Vector3D<site_t>(preambleInfo.BlocksX, 
												   preambleInfo.BlocksY,
												   preambleInfo.BlocksZ),
												   preambleInfo.BlockSize);
		}

		/**
		 * Read the header section, with minimal information about each block.
		 */
		void GeometrySGMYReader::ReadHeader(const SGMY::SGMYPreambleInfo& preambleInfo)
		{
			constexpr size_t nElemPerRead = 4096; // Read Up to 4K Entries at once
			char inbuf[ nElemPerRead * sizeof( SGMY::NonEmptyHeaderRecord) ] __attribute__((aligned(4096)));
			size_t headerOffset = preambleInfo.HeaderOffset;

			const net::MpiCommunicator& comm = file.GetCommunicator();

			size_t recordsRead = 0;
			MPI_Offset fileReadOffset = preambleInfo.HeaderOffset;

	
			while( recordsRead < preambleInfo.NonEmptyBlocks ) {
				size_t blocksToReadThisRound = preambleInfo.NonEmptyBlocks - recordsRead;
				if( blocksToReadThisRound > nElemPerRead ) blocksToReadThisRound = nElemPerRead;

			    file.ReadAt(fileReadOffset, &inbuf[0], 
								blocksToReadThisRound*sizeof(SGMY::NonEmptyHeaderRecord));

			    recordsRead += blocksToReadThisRound;
				fileReadOffset += blocksToReadThisRound*sizeof(SGMY::NonEmptyHeaderRecord);

				// Now this chunk is read, process it out		
           		SGMY::NonEmptyHeaderRecord* headerBlocks = reinterpret_cast<SGMY::NonEmptyHeaderRecord *>(&inbuf[0]);
				for(size_t i=0; i < blocksToReadThisRound; i++) { 
					size_t block = headerBlocks[i].blockNumber;

					blockFileOffsets[block] = headerBlocks[i].fileOffset;
					blockInformation[block].first  = headerBlocks[i].bytes;
					blockInformation[block].second = headerBlocks[i].uncompressedBytes;
					fluidSitesOnEachBlock[block] = headerBlocks[i].sites;
#ifdef HEMELB_USE_GMYPLUS
 				    // 'Computational weight' of this block.
					blockWeights[block] = headerBlocks[i].weights;
#endif	
				}
			}
			
		}

		/**
		 * Read in the necessary blocks from the file.
		 */
		void GeometrySGMYReader::ReadInBlocksWithHalo(Geometry& geometry,
				std::unordered_map<site_t, proc_t>& unitForEachBlock,
				std::unordered_map<site_t, proc_t>& unitForEachBlockFiltered,
				const proc_t localRank, const SGMY::SGMYPreambleInfo& preambleInfo )
		{
			// Create a list of which blocks to read in.
			timings[hemelb::reporting::Timers::readBlocksPrelim].Start();

			// Populate the list of blocks to read (including a halo one block wide around all
			// local blocks).
			log::Logger::Log<log::Debug, log::OnePerCore>("----> determining blocks to read");
			std::vector<site_t> readBlocksArray;
			{

			   std::unordered_set<site_t> readBlock = DecideWhichBlocksToReadIncludingHalo(geometry,
					unitForEachBlock,
					unitForEachBlockFiltered,
					localRank);

			   readBlocksArray.reserve(readBlock.size());
				
			   // Iterate through the blocks to read
			   for(auto mapIter=readBlock.begin(); mapIter != readBlock.end(); ++mapIter) { 
				 site_t nextBlockToRead = *mapIter;

			     // Filter only nonempty blocks
				 if( blockInformation.find(nextBlockToRead) != blockInformation.end()) {
				   readBlocksArray.push_back(nextBlockToRead); 
				 }
               }  
			}

			unitForEachBlock.clear();

			timings[hemelb::reporting::Timers::readBlocksPrelim].Stop();

			// Set the initial offset to the first block, which will be updated as we progress
			// through the blocks.
			MPI_Offset baseOffset = preambleInfo.DataOffset;

			log::Logger::Log<log::Debug, log::OnePerCore>("----> ReadInBlocks() (start)");
			timings[hemelb::reporting::Timers::readBlocksAll].Start();

   			int maxNThreads = omp_get_max_threads();
			std::vector< std::unordered_map<site_t, BlockReadResult> > maps(maxNThreads);
         
			const size_t upper = readBlocksArray.size();

#pragma omp parallel for schedule(static,1)
			for( size_t i = 0; i < upper; i++  ) {

			  int tid = omp_get_thread_num();

			  std::unordered_map<site_t, BlockReadResult>& myMap = maps[tid];

			  site_t nextBlockToRead = readBlocksArray[i];

			  MPI_Offset fileOffset = baseOffset + blockFileOffsets[nextBlockToRead];
			  auto nBytes = blockInformation.at(nextBlockToRead).first;

			  // Read data
			  std::vector<char> compressedBlockData(nBytes);
		  	  file.ReadAt(fileOffset, compressedBlockData);

			  // Decompress and Parse
			  std::vector<char> blockData = DecompressBlockData(compressedBlockData,
												blockInformation.at(nextBlockToRead).second);
 		      io::writers::xdr::XdrMemReader lReader(&blockData.front(), blockData.size());

			  BlockReadResult theBlock;
			  theBlock.Sites.clear();
			  ParseBlock(theBlock,geometry.GetSitesPerBlock(), lReader);
			  myMap[nextBlockToRead] = theBlock;
			}

			// Now merge the maps
			for(int i=0; i < maxNThreads; i++) {
			    if( maps[i].size() > 0) {
					geometry.Blocks.insert( maps[i].begin(), maps[i].end());
				}
			}
			// In the regular read, readBlock() and blockInformation would clear
			blockFileOffsets.clear();

				
			// blockInformation.clear(); -- don't clear this we can use it in optimizing the decomposition.

			timings[hemelb::reporting::Timers::readBlocksAll].Stop();
			log::Logger::Log<log::Debug, log::OnePerCore>("----> ReadInBlocks() (end)");
			
		}

		void GeometrySGMYReader::ReadInBlock(MPI_Offset offsetSoFar, Geometry& geometry,
				const std::vector<proc_t>& procsWantingThisBlock,
				const proc_t readingCore, const site_t blockNumber, const bool neededOnThisRank)
		{
			// Easy case if there are no sites on the block.
			if (blockInformation.find(blockNumber) == blockInformation.end())
			{
				return;
			}

			net::Net net = net::Net(computeComms);

			std::vector<char> compressedBlockData;
			if (readingCore == computeComms.Rank())
			{
				log::Logger::Log<log::Debug, log::OnePerCore>("------> blockNumber = %li with:\n"
						"bytes             = %u,\n"
						"uncompressedBytes = %u and offsetSoFar = %li",
						blockNumber,
						blockInformation.at(blockNumber).first,
						blockInformation.at(blockNumber).second,
						offsetSoFar);

				timings[hemelb::reporting::Timers::readBlock].Start();
				// Read the data.
				compressedBlockData.resize(blockInformation.at(blockNumber).first);
				file.ReadAt(offsetSoFar, compressedBlockData);

				// Spread it...
				// unless procsWantingBlocksBuffer (procsWantingThisBlock) is empty.
				if (procsWantingThisBlock.front() != -1) {
					for (std::vector<proc_t>::const_iterator receiver = procsWantingThisBlock.begin(); receiver
							!= procsWantingThisBlock.end(); receiver++)
					{
						if (*receiver != computeComms.Rank())
						{
#ifdef HEMELB_USE_MPI_CALL
							MPI_Send(&compressedBlockData[0],
									compressedBlockData.size(),
									MPI_CHAR, *receiver, 0, computeComms);
#else
							net.RequestSendV(compressedBlockData, *receiver);
#endif
						}
					}
					timings[hemelb::reporting::Timers::readBlock].Stop();
				}
			}
			else if (neededOnThisRank)
			{
				compressedBlockData.resize(blockInformation.at(blockNumber).first);
#ifdef HEMELB_USE_MPI_CALL
				MPI_Recv(&compressedBlockData[0],
						compressedBlockData.size(),
						MPI_CHAR, readingCore, 0, computeComms, MPI_STATUS_IGNORE);
#else
				net.RequestReceiveV(compressedBlockData, readingCore);
#endif
			}
			else
			{
				return;
			}
			timings[hemelb::reporting::Timers::readNet].Start();
#ifndef HEMELB_USE_MPI_CALL
			net.Dispatch();
#endif
			timings[hemelb::reporting::Timers::readNet].Stop();

			timings[hemelb::reporting::Timers::readParse].Start();
			if (neededOnThisRank)
			{
				// Create an Xdr interpreter.
				std::vector<char> blockData = DecompressBlockData(compressedBlockData,
						blockInformation.at(blockNumber).second);
				io::writers::xdr::XdrMemReader lReader(&blockData.front(), blockData.size());

				ParseBlock(geometry, blockNumber, lReader);
			}
			//else if (geometry.Blocks.find(blockNumber) != geometry.Blocks.end())
			//{
			//	geometry.Blocks[blockNumber].Sites = std::vector<GeometrySite>(0, GeometrySite(false));
			//}
			timings[hemelb::reporting::Timers::readParse].Stop();
		}

		std::vector<char> GeometrySGMYReader::DecompressBlockData(const std::vector<char>& compressed,
				const unsigned int uncompressedBytes)
		{
			timings[hemelb::reporting::Timers::unzip].Start();
			// For zlib return codes.
			int ret;

			// Set up the buffer for decompressed data.
			std::vector<char> uncompressed(uncompressedBytes);

			// Set up the inflator.
			z_stream stream;
			stream.zalloc = Z_NULL;
			stream.zfree = Z_NULL;
			stream.opaque = Z_NULL;
			stream.avail_in = compressed.size();
			stream.next_in = reinterpret_cast<unsigned char*> (const_cast<char*> (&compressed.front()));

			ret = inflateInit(&stream);
			if (ret != Z_OK)
				throw Exception() << "Decompression error for block!";

			stream.avail_out = uncompressed.size();
			stream.next_out = reinterpret_cast<unsigned char*> (&uncompressed.front());

			ret = inflate(&stream, Z_FINISH);
			if (ret != Z_STREAM_END)
				throw Exception() << "Decompression error for block!";

			uncompressed.resize(uncompressed.size() - stream.avail_out);
			ret = inflateEnd(&stream);
			if (ret != Z_OK)
				throw Exception() << "Decompression error for block!";

			timings[hemelb::reporting::Timers::unzip].Stop();
			return uncompressed;
		}


		void GeometrySGMYReader::ParseBlock(BlockReadResult& theBlock, site_t sitesPerBlock,
				io::writers::xdr::XdrReader& reader)
		{
			// We start by clearing the sites on the block. We read the blocks twice (once before
			// optimisation and once after), so there can be sites on the block from the previous read.
			for (site_t localSiteIndex = 0; localSiteIndex < sitesPerBlock; ++localSiteIndex)
			{
				theBlock.Sites.push_back(  ParseSite(reader) );
			}
		}

		void GeometrySGMYReader::ParseBlock(Geometry& geometry, const site_t block,
				io::writers::xdr::XdrReader& reader)
		{
			// We start by clearing the sites on the block. We read the blocks twice (once before
			// optimisation and once after), so there can be sites on the block from the previous read.
			geometry.Blocks[block].Sites.clear();

			for (site_t localSiteIndex = 0; localSiteIndex < geometry.GetSitesPerBlock(); ++localSiteIndex)
			{
				geometry.Blocks.at(block).Sites.push_back(ParseSite(reader));
			}
		}

		GeometrySite GeometrySGMYReader::ParseSite(io::writers::xdr::XdrReader& reader)
		{
			// Read the fluid property.
			unsigned isFluid;
			bool success = reader.readUnsignedInt(isFluid);

			if (!success)
			{
				log::Logger::Log<log::Error, log::OnePerCore>("Error reading site type!");
			}

			/// @todo #598 use constant in hemelb::io::formats::geometry
			GeometrySite readInSite(isFluid != 0);

			// If solid, there's nothing more to do.
			if (!readInSite.isFluid)
			{
				return readInSite;
			}

			const io::formats::geometry::DisplacementVector& neighbourhood =
				io::formats::geometry::Get().GetNeighbourhood();
			// Prepare the links array to have enough space.
			readInSite.links.resize(latticeInfo.GetNumVectors() - 1);

			bool isGmyWallSite = false;

			// For each link direction...
			for (Direction readDirection = 0; readDirection < neighbourhood.size(); readDirection++)
			{
				// read the type of the intersection and create a link...
				unsigned intersectionType;
				reader.readUnsignedInt(intersectionType);

				GeometrySiteLink link;
				link.type = (GeometrySiteLink::IntersectionType) intersectionType;

				// walls have a floating-point distance to the wall...
				if (link.type == GeometrySiteLink::WALL_INTERSECTION)
				{
					isGmyWallSite = true;
					float distance;
					reader.readFloat(distance);
					link.distanceToIntersection = distance;
				}

				// inlets and outlets (which together with none make up the other intersection types)
				// have an iolet id and a distance float...
				else if (link.type != GeometrySiteLink::NO_INTERSECTION)
				{
					float distance;
					unsigned ioletId;
					reader.readUnsignedInt(ioletId);
					reader.readFloat(distance);

					link.ioletId = ioletId;
					link.distanceToIntersection = distance;
				}

				// now, attempt to match the direction read from the local neighbourhood to one in the
				// lattice being used for simulation. If a match is found, assign the link to the read
				// site.
				for (Direction usedLatticeDirection = 1; usedLatticeDirection < latticeInfo.GetNumVectors(); usedLatticeDirection++)
				{
					if (latticeInfo.GetVector(usedLatticeDirection) == neighbourhood[readDirection])
					{
						// If this link direction is necessary to the lattice in use, keep the link data.
						readInSite.links[usedLatticeDirection - 1] = link;
						break;
					}
				}
			}

			unsigned normalAvailable;
			reader.readUnsignedInt(normalAvailable);
			readInSite.wallNormalAvailable = (normalAvailable
					== io::formats::geometry::WALL_NORMAL_AVAILABLE);

			if (readInSite.wallNormalAvailable != isGmyWallSite)
			{
				std::string msg = isGmyWallSite
					? "wall fluid site without"
					: "bulk fluid site with";
				throw Exception() << "Malformed GMY file, "
					<< msg << " a defined wall normal currently not allowed.";
			}

			if (readInSite.wallNormalAvailable)
			{
				reader.readFloat(readInSite.wallNormal[0]);
				reader.readFloat(readInSite.wallNormal[1]);
				reader.readFloat(readInSite.wallNormal[2]);
			}

			return readInSite;
		}

		proc_t GeometrySGMYReader::GetReadingCoreForBlock(site_t blockNumber)
		{
			return proc_t(blockNumber % util::NumericalFunctions::min(READING_GROUP_SIZE,
						computeComms.Size()));
		}

		std::unordered_set<site_t> GeometrySGMYReader::DecideWhichBlocksToReadIncludingHalo(
				const Geometry& geometry,
				const std::unordered_map<site_t, proc_t>& unitForEachBlock,
				std::unordered_map<site_t, proc_t>& unitForEachBlockFiltered,
				proc_t localRank)
		{
			std::unordered_set<site_t> shouldReadBlock;

			// Read a block in if it has fluid sites and is to live on the current processor. Also read
			// in any neighbours with fluid sites.
			for (site_t blockI = 0; blockI < geometry.GetBlockDimensions().x; ++blockI)
			{
				for (site_t blockJ = 0; blockJ < geometry.GetBlockDimensions().y; ++blockJ)
				{
					for (site_t blockK = 0; blockK < geometry.GetBlockDimensions().z; ++blockK)
					{
						site_t lBlockId = geometry.GetBlockIdFromBlockCoordinates(blockI, blockJ, blockK);

						if (unitForEachBlock.find(lBlockId) == unitForEachBlock.end())
						{
							continue;
						}

						if (unitForEachBlock.at(lBlockId) != localRank)
						{
							continue;
						}

						// Read in all neighbouring blocks.
						for (site_t neighI = util::NumericalFunctions::max<site_t>(0, blockI - 1); (neighI
									<= (blockI + 1)) && (neighI < geometry.GetBlockDimensions().x); ++neighI)
						{
							for (site_t neighJ = util::NumericalFunctions::max<site_t>(0, blockJ - 1); (neighJ
										<= (blockJ + 1)) && (neighJ < geometry.GetBlockDimensions().y); ++neighJ)
							{
								for (site_t neighK = util::NumericalFunctions::max<site_t>(0, blockK - 1); (neighK
											<= (blockK + 1)) && (neighK < geometry.GetBlockDimensions().z); ++neighK)
								{
									site_t lNeighId = geometry.GetBlockIdFromBlockCoordinates(neighI, neighJ, neighK);
									if (unitForEachBlock.find(lNeighId) != unitForEachBlock.end())
										unitForEachBlockFiltered[lNeighId] = unitForEachBlock.at(lNeighId);
									shouldReadBlock.insert(lNeighId);
								}
							}
						}
					}
				}
			}
			return shouldReadBlock;
		}

		void GeometrySGMYReader::OptimiseDomainDecomposition(Geometry& geometry,
				std::unordered_map<site_t, proc_t>& procForEachBlock,
				std::unordered_map<site_t, proc_t>& procForEachBlockFiltered)
		{

			decomposition::OptimisedDecomposition optimiser(timings,
					computeComms,
					geometry,
					latticeInfo,
					procForEachBlock,
					fluidSitesOnEachBlock);

			site_t geometrySize = geometry.Blocks.size();

			timings[hemelb::reporting::Timers::moves].Start();
			
			// Implement the decomposition now that we have read the necessary data.
			ImplementMoves(geometry,
					procForEachBlock,
					procForEachBlockFiltered,
					optimiser.GetMovesCountPerCore(),
					optimiser.GetMovesList());
			timings[hemelb::reporting::Timers::moves].Stop();

			// BJ: I don't think this is really necessarty -- it writes the block mapping
			// log::Logger::Log<log::Info, log::OnePerCore>("------> show decomposition");
			// ShowDecomposition(geometry, procForEachBlockFiltered);
		}

		void GeometrySGMYReader::ShowDecomposition(Geometry& geometry,
				const std::unordered_map<site_t, proc_t>& procForEachBlockFiltered) const
		{
			// Open file for writing.
			std::ofstream myfile;
			myfile.open("./decomposition/globalSiteCoords_" + std::to_string(computeComms.Rank()) + ".txt");

			const int blockSize = geometry.GetBlockSize();
			for (site_t blockI = 0; blockI < geometry.GetBlockDimensions().x; blockI++)
				for (site_t blockJ = 0; blockJ < geometry.GetBlockDimensions().y; blockJ++)
					for (site_t blockK = 0; blockK < geometry.GetBlockDimensions().z; blockK++)
					{
						const site_t blockNumber = geometry.GetBlockIdFromBlockCoordinates(
								blockI,
								blockJ,
								blockK);

						// Does block contain fluid sites?
						if (procForEachBlockFiltered.find(blockNumber) == procForEachBlockFiltered.end())
							continue;
						// Is block owned by this rank?
						if (procForEachBlockFiltered.at(blockNumber) != computeComms.Rank())
							continue;

						const BlockReadResult& blockReadResult = geometry.Blocks.at(blockNumber);

						//site_t blockIJData = blockNumber / geometry.GetBlockDimensions().z;
						//blockCoords.x = blockIJData / geometry.GetBlockDimensions().y;
						//blockCoords.y = blockIJData % geometry.GetBlockDimensions().y;
						//blockCoords.z = blockNumber % geometry.GetBlockDimensions().z;

						site_t m = 0;
						// Iterate over sites within the block.
						for (site_t localSiteI = 0; localSiteI < blockSize; localSiteI++)
							for (site_t localSiteJ = 0; localSiteJ < blockSize; localSiteJ++)
								for (site_t localSiteK = 0; localSiteK < blockSize; localSiteK++)
								{
									//site_t localSiteID =
									//	geometry.GetSiteIdFromSiteCoordinates(localSiteI, localSiteJ, localSiteK);
									//site_t siteIJData = localSiteID / block_size;
									//localSiteCoords.x = siteIJData  / block_size;
									//localSiteCoords.y = siteIJData  % block_size;
									//localSiteCoords.z = localSiteID % block_size;
									//util::Vector3D<site_t> globalSiteCoords = blockCoords * blockSize + localSiteCoords;

								  //if (blockReadResult.Sites[m].isFluid)
									if (blockReadResult.Sites[m].targetProcessor != SITE_OR_BLOCK_SOLID)
										myfile <<
											blockReadResult.Sites[m].isFluid << " " <<
											blockReadResult.Sites[m].targetProcessor << " " <<
											blockNumber << " " <<
											blockI * blockSize + localSiteI << " " <<
											blockJ * blockSize + localSiteJ << " " <<
											blockK * blockSize + localSiteK << std::endl; m++;
								}
					} myfile.close();
		}

		// The header section of the config file contains a number of records.
		site_t GeometrySGMYReader::GetHeaderLength(site_t blockCount) const
		{
			return io::formats::geometry::HeaderRecordLength * blockCount;
		}

		void GeometrySGMYReader::ImplementMoves(Geometry& geometry,
				const std::unordered_map<site_t, proc_t>& procForEachBlock,
				const std::unordered_map<site_t, proc_t>& procForEachBlockFiltered,
				const std::vector<idx_t>& movesFromEachProc,
				const std::vector<idx_t>& movesList) const
		{

			log::Logger::Log<log::Debug, log::OnePerCore>("----> ImplementMoves(): procForEachBlockFiltered.size() == %i", procForEachBlockFiltered.size());
			// First all, set the proc rank for each site to what it originally was before
			// domain decomposition optimisation. Go through each block.
			for (auto& kv : geometry.Blocks)
			{
				site_t block = kv.first;
				BlockReadResult& currentBlock = kv.second;

				// If this proc has owned a fluid site on this block either before or after optimisation,
				// the following will be non-null.
				// if (geometry.Blocks.find(block) != geometry.Blocks.end())
				// {
					// Get the original proc for that block.
					proc_t originalProc = procForEachBlockFiltered.at(block);

					// For each site on that block...
					#pragma omp parallel for schedule(static,1)
					for (site_t siteIndex = 0; siteIndex < geometry.GetSitesPerBlock(); ++siteIndex)
					{
						// if the site is non-solid...
						if (currentBlock.Sites[siteIndex].targetProcessor != SITE_OR_BLOCK_SOLID)
						{
							// set its rank to be the rank it had before optimisation.
							currentBlock.Sites[siteIndex].targetProcessor
								= ConvertTopologyRankToGlobalRank(originalProc);
						}
					}
				// }
			}

		}

		proc_t GeometrySGMYReader::ConvertTopologyRankToGlobalRank(proc_t topologyRankIn) const
		{
			// If the global rank is not equal to the topology rank, we are not using rank 0 for
			// LBM.
			return (hemeLbComms.Rank() == computeComms.Rank())
				? topologyRankIn
				: (topologyRankIn + 1);
		}

		bool GeometrySGMYReader::ShouldValidate() const
		{
#ifdef HEMELB_VALIDATE_GEOMETRY
			return true;
#else
			return false;
#endif
		}
	}
}
