
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_COLLOIDS_COLLOIDCONTROLLER_H
#define HEMELB_COLLOIDS_COLLOIDCONTROLLER_H

#include <vector>
#include "net/net.h"
#include "net/IteratedAction.h"
#include "geometry/LatticeData.h"
#include "geometry/Geometry.h"
#include "io/xml/XmlAbstractionLayer.h"
#include "lb/MacroscopicPropertyCache.h"
#include "colloids/ParticleSet.h"
#include "util/Vector3D.h"
#include "units.h"

namespace hemelb
{
	namespace colloids
	{
		/** provides the control interface between colloid simulation and the rest of the system */
		class ColloidController : public net::IteratedAction
		{
			public:
				/** constructor - currently only initialises the neighbour list */
				ColloidController(geometry::LatticeData& latDatLBM,
						const lb::SimulationState& simulationState,
						const configuration::SimConfig* simConfig,
						const geometry::Geometry& gmyResult,
						io::xml::Document& xml,
						lb::MacroscopicPropertyCache& propertyCache,
						//const hemelb::lb::LbmParameters *lbmParams,
						const std::string& outputPath,
						const net::IOCommunicator& ioComms_,
						reporting::Timers& timers);

				/** destructor - releases resources allocated by this class */
				~ColloidController();

				/** overloaded from IteratedAction */
				void RequestComms();

				/** overloaded from IteratedAction */
				void EndIteration();

				const void OutputInformation(const LatticeTimeStep timestep) const;

			private:
				/** Main code communicator */
				const net::IOCommunicator& ioComms;

				const lb::SimulationState& simulationState;

				/** holds the set of Particles that this processor knows about */
				ParticleSet* particleSet;

				/** Timers object, for generating timing data for reports.*/
				reporting::Timers& timers;

				/** maximum separation from a colloid of sites used in its fluid velocity interpolation */
				const static site_t REGION_OF_INFLUENCE = (site_t)2;

				/** particles emitted every emissionItrvl steps
				    particle emission off if emissionItrvl == 0 */
				unsigned emissionItrvl;

				/** is particle emission enabled? */
				bool emissionFlag;

				/** a vector of the processors that might be interested in
				    particles near the edge of this processor's sub-domain */
				std::vector<proc_t> neighbourProcessors;

				/** a list of relative 3D vectors that defines the sites within a region of influence */
				typedef std::vector<util::Vector3D<site_t> > Neighbourhood;

				/** obtains the neighbourhood for a particular region of influence defined by distance */
				const Neighbourhood GetNeighbourhoodVectors(site_t distance);

				/** determines the list of neighbour processors
				  i.e. processors that are within the region of influence of the local domain's edge
				  i.e. processors that own at least one site in the neighbourhood of a local site */
				void InitialiseNeighbourList(const geometry::LatticeData& latDatLBM,
						const geometry::Geometry& gmyResult,
						const Neighbourhood& neighbourhood);

				/** get local coordinates and the owner rank for a site from its global coordinates */
				bool GetLocalInformationForGlobalSite(const geometry::Geometry& gmyResult,
						const util::Vector3D<site_t>& globalLocationForSite,
						site_t* blockIdForSite,
						site_t* localSiteIdForSite,
						proc_t* ownerRankForSite);
		};
	}
}

#endif /* HEMELB_COLLOIDS_COLLOIDCONTROLLER */
