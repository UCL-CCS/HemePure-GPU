
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_GEOMETRY_GEOMETRYSGMYREADER_H
#define HEMELB_GEOMETRY_GEOMETRYSGMYREADER_H

#include <vector>
#include <string>

#include <unordered_map>
#include <unordered_set>

#include "io/writers/xdr/XdrReader.h"
#include "io/formats/formats.h"
#include "lb/lattices/LatticeInfo.h"
#include "lb/LbmParameters.h"
#include "net/mpi.h"
#include "geometry/ParmetisHeader.h"
#include "reporting/Timers.h"
#include "util/Vector3D.h"
#include "units.h"
#include "geometry/Geometry.h"
#include "geometry/needs/Needs.h"

#include "net/MpiFile.h"


namespace hemelb
{
	namespace geometry
	{
		namespace SGMY {

			constexpr uint32_t SgmyMagicNumber = 0x676d7905;
			constexpr uint32_t SgmyVersionNumber = 1;

			struct NonEmptyHeaderRecord {
			  uint64_t   blockNumber;  // 8 bytes
			  uint64_t   fileOffset;   // 8 bytes
			  uint32_t sites;          // 4 bytes
			  uint32_t bytes;           // 4 bytes
			  uint32_t uncompressedBytes; // 4 bytes  Total 28 bytes
			  uint32_t weights;			  // 4 bytes  Total 32 bytes
			};


			struct SGMYPreambleInfo {
			  uint32_t HemeLBMagic;
			  uint32_t SgmyMagic;

			  uint32_t Version;
			  uint32_t BlocksX;

			  uint32_t BlocksY;
			  uint32_t BlocksZ;

			  uint32_t BlockSize;
			  uint32_t MaxCompressedBytes;

			  uint32_t MaxUncompressedBytes;
			  uint32_t HeaderOffset;

			  uint64_t NonEmptyBlocks;
			  uint64_t DataOffset;
			  SGMYPreambleInfo() : HemeLBMagic(hemelb::io::formats::HemeLbMagicNumber),
									 SgmyMagic(SgmyMagicNumber),
									 Version(SgmyVersionNumber),
									 BlocksX(0), BlocksY(0), BlocksZ(0),
									 MaxCompressedBytes(0), MaxUncompressedBytes(0),
									 HeaderOffset(56),
									 NonEmptyBlocks(0),
									 DataOffset(0) {}

			};
		} // namespace SGMY


		class GeometrySGMYReader
		{
			public:
				typedef util::Vector3D<site_t> BlockLocation;

				GeometrySGMYReader(const lb::lattices::LatticeInfo&,
						reporting::Timers &timings, const net::IOCommunicator& ioComm);
				~GeometrySGMYReader();

				Geometry LoadAndDecompose(const std::string& dataFilePath);

			private:

				/**
				 * Read from the file into a buffer. We read this on a single core then broadcast it.
				 * This has proven to be more efficient than reading in on every core (even using a collective
				 * read).
				 *
				 * Note this allocates memory and returns the pointer to you.
				 *
				 * @param nBytes
				 * @return
				 */
				std::vector<char> ReadOnAllTasks(sitedata_t nBytes);


				/** 
                 * Reads the SGMY Preamble. It returns a base initialized Geometry to Us
				 * and also an SGMY PreambleInfo structure.
				 */ 	
				Geometry ReadPreamble(SGMY::SGMYPreambleInfo& preambleInfo);

				void ReadHeader(const SGMY::SGMYPreambleInfo& preambleInfo);

				void ReadInBlocksWithHalo(Geometry& geometry,
						std::unordered_map<site_t, proc_t>& unitForEachBlock,
						std::unordered_map<site_t, proc_t>& unitForEachBlockFiltered,
						const proc_t localRank, const SGMY::SGMYPreambleInfo& preambleInfo);

				/**
				 * Compile a list of blocks to be read onto this core, including all the ones we perform
				 * LB on, and also any of their neighbouring blocks.
				 *
				 * NOTE: that the skipping-over of blocks without any fluid sites is dealt with by other
				 * code.
				 *
				 * @param geometry [in] Geometry object as it has been read so far
				 * @param unitForEachBlock [in] The initial processor assigned to each block
				 * @param localRank [in] Local rank number
				 * @return Vector with true for each block we should read in.
				 */
				std::unordered_set<site_t> DecideWhichBlocksToReadIncludingHalo(const Geometry& geometry,
						const std::unordered_map<site_t, proc_t>& unitForEachBlock,
						std::unordered_map<site_t, proc_t>& unitForEachBlockFiltered,
						const proc_t localRank);

				/**
				 * Reads in a single block and ensures it is distributed to all cores that need it.
				 *
				 * @param offsetSoFar [in] The offset into the file to read from to get the block.
				 * @param geometry [out] The geometry object to populate with info about the block.
				 * @param procsWantingThisBlock [in] A list of proc ids where info about this block is required.
				 * @param blockNumber [in] The id of the block we're reading.
				 * @param neededOnThisRank [in] A boolean indicating whether the block is required locally.
				 */
				void ReadInBlock(MPI_Offset offsetSoFar,
						Geometry& geometry,
						const std::vector<proc_t>& procsWantingThisBlock,
						const proc_t readingCore,
						const site_t blockNumber,
						const bool neededOnThisRank);

				/**
				 * Decompress the block data. Uses the known number of sites to get an
				 * upper bound on the uncompressed data to simplify the code and avoid
				 * reallocation.
				 * @param compressed
				 * @param sites
				 * @return
				 */
				std::vector<char> DecompressBlockData(const std::vector<char>& compressed,
						const unsigned int uncompressedBytes);


				void ParseBlock(BlockReadResult& theBlock, site_t sitesPerBlock, io::writers::xdr::XdrReader& reader);
				void ParseBlock(Geometry& geometry, const site_t block, io::writers::xdr::XdrReader& reader);

				/**
				 * Parse the next site from the XDR reader. Note that we return by copy here.
				 * @param reader
				 * @return
				 */
				GeometrySite ParseSite(io::writers::xdr::XdrReader& reader);

				/**
				 * Calculates the number of the rank used to read in a given block.
				 * Intent is to move this into Decomposition class, which will also handle knowledge of which procs to use for reading, and own the decomposition topology.
				 *
				 * @param blockNumber
				 * @return
				 */
				proc_t GetReadingCoreForBlock(site_t blockNumber);

				/**
				 * Optimise the domain decomposition using ParMetis. We take this approach because ParMetis
				 * is more efficient when given an initial decomposition to start with.
				 * @param geometry
				 * @param procForEachBlock
				 */
				void OptimiseDomainDecomposition(Geometry& geometry,
						std::unordered_map<site_t, proc_t>& procForEachBlock,
						std::unordered_map<site_t, proc_t>& procForEachBlockFiltered);

				void ValidateGeometry(const Geometry& geometry);

				/**
				 * Get the length of the header section, given the number of blocks.
				 *
				 * @param blockCount The number of blocks.
				 * @return
				 */
				site_t GetHeaderLength(site_t blockCount) const;

				void RereadBlocks(Geometry& geometry,
						const std::vector<idx_t>& movesPerProc,
						const std::vector<idx_t>& movesList,
						std::unordered_map<site_t, proc_t>& procForEachBlock,
						std::unordered_map<site_t, proc_t>& procForEachBlockFiltered);

				void ImplementMoves(Geometry& geometry,
						const std::unordered_map<site_t, proc_t>& procForEachBlock,
						const std::unordered_map<site_t, proc_t>& procForEachBlockFiltered,
						const std::vector<idx_t>& movesFromEachProc,
						const std::vector<idx_t>& movesList) const;

				void ShowDecomposition(Geometry& geometry,
						const std::unordered_map<site_t, proc_t>& procForEachBlockFiltered) const;

				proc_t ConvertTopologyRankToGlobalRank(proc_t topologyRank) const;

				/**
				 * True if we should validate the geometry.
				 * @return
				 */
				bool ShouldValidate() const;

				//! The rank which reads in the header information.
				static const proc_t HEADER_READING_RANK = 0;
				//! The number of cores (0 to READING_GROUP_SIZE-1) that read files in parallel.
				static const proc_t READING_GROUP_SIZE = HEMELB_READING_GROUP_SIZE;
				//! The spacing between cores that read files in parallel.
				static const proc_t READING_GROUP_SPACING = HEMELB_READING_GROUP_SPACING;

				//! Info about the connectivity of the lattice.
				const lb::lattices::LatticeInfo& latticeInfo;
				
				net::MpiFile file;

				//! HemeLB's main communicator.
				const net::IOCommunicator& hemeLbComms;
				//! Communication info for all ranks that will need a slice of the geometry.
				net::MpiCommunicator computeComms;
				//! True if this rank is participating in the domain decomposition.
				bool participateInTopology;

				//! The offset for each block in the file (after preamble and header)
				std::unordered_map<site_t, size_t> blockFileOffsets;

				//! The number of non-empty blocks in the geometry.
				sitedata_t nonEmptyBlocks;
				//! Essential block information:
				//!		the number of bytes each block takes up while still compressed.
				//!		the number of bytes each block takes up when uncompressed.
				std::unordered_map<site_t, std::pair<uint32_t, uint32_t> > blockInformation;
				//! The number of fluid sites on each block in the geometry.
				std::unordered_map<site_t, uint32_t> fluidSitesOnEachBlock;
				//! Block weights.
				std::unordered_map<site_t, uint32_t> blockWeights;

				//! Timings object for recording the time taken for each step of the domain decomposition.
				hemelb::reporting::Timers &timings;
		};
	}
}

#endif /* HEMELB_GEOMETRY_GEOMETRYREADER_H */
