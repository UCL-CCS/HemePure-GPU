
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_GEOMETRY_DECOMPOSITION_BASICDECOMPOSITION_H
#define HEMELB_GEOMETRY_DECOMPOSITION_BASICDECOMPOSITION_H

#include <unordered_map>
#include <unordered_set>

//#include "ALL.hpp"
//#include "ALL_Point.hpp"
//#include "ALL_CustomExceptions.hpp"

#include "geometry/Geometry.h"
#include "lb/lattices/LatticeInfo.h"
#include "net/MpiCommunicator.h"
#include "units.h"
#include "util/Vector3D.h"

namespace hemelb
{
	namespace geometry
	{
		namespace decomposition
		{
			class BasicDecomposition
			{
				public:
					typedef util::Vector3D<site_t> BlockLocation;

					/**
					 * Constructor to populate all fields necessary for a decomposition
					 *
					 * @param geometry
					 * @param communicator
					 * @param blockInformation
					 * @param blockWeights
					 */
					BasicDecomposition(const Geometry& geometry,
							const lb::lattices::LatticeInfo& latticeInfo,
							const net::MpiCommunicator& communicator,
							const std::unordered_map<site_t, std::pair<uint16_t, uint16_t> >& blockInformation,
							const std::unordered_map<site_t, uint16_t>& blockWeights);

					/**
					 * Does a basic decomposition of the geometry without requiring any communication;
					 * produces a vector of the processor assigned to each block.
					 *
					 * To make this fast and remove a need to read in the site info about blocks,
					 * we assume that neighbouring blocks with any fluid sites on have lattice links
					 * between them.
					 *
					 * NOTE that the old version of this code used to cope with running on multiple machines,
					 * by decomposing fluid sites over machines, then decomposing over processors within one machine.
					 * To achieve this here, one could use parmetis's "nparts" parameters to first decompose over machines, then
					 * to decompose within machines.
					 *
					 * @param rankForEachBlock A vector with the processor rank each block has been assigned to.
					 */
					void Decompose(
							std::unordered_map<site_t, proc_t>& procAssignedToEachBlock);

					/**
					 * Does a basic decomposition of the geometry without requiring any communication;
					 * produces a vector of the processor assigned to each block.
					 *
					 * @param rankForEachBlock A vector with the processor rank each block has been assigned to.
					 */
					void DecomposeDumb(
							std::unordered_map<site_t, proc_t>& procAssignedToEachBlock,
							sitedata_t nonEmptyBlocks);
					//void DecomposeBlock(
					//		std::unordered_map<site_t, proc_t>& procAssignedToEachBlock, int noderank);

					//void RotateAndAllocate(
					//		const double (&m_rot)[3][3][3], const double (&phi)[3],
					//		const double (&r_min)[3], const double (&r_max)[3],
					//		std::vector<ALL_Point<double>>& points,
					//		std::unordered_map<site_t, proc_t>& procAssignedToEachBlock,
					//		const std::vector<double>& l, const std::vector<double>& u,
					//		const int rank);

					/**
					 * Validates that all cores have the same beliefs about which proc is to be assigned
					 * to each proc by this decomposition.
					 *
					 * @param procAssignedToEachBlock This core's decomposition result.
					 */
					void Validate(std::unordered_map<site_t, proc_t>& procAssignedToEachBlock);

				private:

					void Procs2Grid(int nx, int ny, int nz, int &px, int &py, int &pz);

					/**
					 * Does the work of dividing blocks up between processors.
					 *
					 * The algorithm iterates over processors (units).
					 * We start by assigning the next unassigned block to the current unit, then growing out
					 * the region by adding new blocks, until the current unit has approximately the right
					 * number of blocks for the given numbers of blocks / units. When adding blocks, we prefer
					 * blocks that are neighbours of blocks already assigned to the current unit.
					 *
					 * @param unitForEachBlock [out] The processor id for each block
					 * @param unassignedBlocks [in] The number of blocks yet to be assigned a processor
					 * @param geometry [in] The geometry we're decomposing
					 * @param unitCount [in] The total number of processors
					 * @param fluidSitesPerBlock [in] The number of fluid sites in each block
					 * @param blockWeights [in] Block weights
					 */
					void DivideBlocks(std::unordered_map<site_t, proc_t>& unitForEachBlock,
							site_t unassignedBlocks,
							const Geometry& geometry,
							const proc_t unitCount,
							const std::unordered_map<site_t, std::pair<uint16_t, uint16_t> >& blockInformation);

					/**
					 * Attempt to expand an already connected volume of blocks assigned to one processor
					 * to add additional blocks, already connected to the volume.
					 *
					 * Returns true if the region was expanded.
					 *
					 * @param edgeBlocks
					 * @param expansionBlocks
					 * @param blockAssigned
					 * @param currentUnit
					 * @param unitForEachBlock
					 * @param blocksPerUnit
					 * @return Returns true if the region was expanded.
					 */
					bool Expand(std::vector<BlockLocation>& expansionBlocks,
							std::unordered_set<site_t>& blockAssigned,
							std::unordered_map<site_t, proc_t>& unitForEachBlock,
							site_t &blocksOnCurrentUnit,
							const std::vector<BlockLocation>& edgeBlocks,
							const proc_t currentUnit,
							const site_t blocksPerUnit);

					const Geometry& geometry; //! The geometry being decomposed.
					const lb::lattices::LatticeInfo& latticeInfo; //! The lattice to decompose for.
					const net::MpiCommunicator& communicator; //! The communicator object being decomposed over.
					const std::unordered_map<site_t, std::pair<uint16_t, uint16_t> >& blockInformation;
					const std::unordered_map<site_t, uint16_t>& blockWeights;
			};
		} /* namespace decomposition */
	} /* namespace geometry */
} /* namespace hemelb */
#endif /* HEMELB_GEOMETRY_DECOMPOSITION_BASICDECOMPOSITION_H */
