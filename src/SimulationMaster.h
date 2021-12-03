
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_SIMULATIONMASTER_H
#define HEMELB_SIMULATIONMASTER_H
#include "lb/lattices/Lattices.h"
#include "extraction/PropertyActor.h"
#include "lb/lb.hpp"
#include "lb/StabilityTester.h"
#include "net/net.h"
#include "lb/EntropyTester.h"
#include "lb/iolets/BoundaryValues.h"
#include "util/UnitConverter.h"
#include "configuration/CommandLine.h"
#include "io/PathManager.h"
#include "reporting/Reporter.h"
#include "reporting/Timers.h"
#include "reporting/BuildInfo.h"
#include "lb/IncompressibilityChecker.hpp"
#include "colloids/ColloidController.h"
#include "net/phased/StepManager.h"
#include "net/phased/NetConcern.h"
#include "geometry/neighbouring/NeighbouringDataManager.h"


class SimulationMaster
{
  public:
    SimulationMaster(hemelb::configuration::CommandLine &options, const hemelb::net::IOCommunicator& ioComms);
    virtual ~SimulationMaster();

    void Abort();

    bool IsCurrentProcTheIOProc();

    int GetProcessorCount();

    void RunSimulation();
    hemelb::lb::SimulationState const * GetState() const {
      return simulationState;
    }
    void Finalise();

#ifdef HEMELB_USE_GPU
	void check_GPU_capabilities();
	// void FinaliseGPU();
	// Should Abort if no GPUs are found - or change to CPU only sims
#endif


  protected:

    hemelb::lb::iolets::BoundaryValues* inletValues;
    hemelb::lb::iolets::BoundaryValues* outletValues;
    virtual void DoTimeStep();

    /* The next quantities are protected because they are used by MultiscaleSimulationMaster */
    // Set the lattice type via a build parameter
    typedef hemelb::lb::lattices:: HEMELB_LATTICE latticeType;
    hemelb::geometry::LatticeData* latticeData;
    hemelb::lb::LBM<latticeType>* latticeBoltzmannModel;
    hemelb::geometry::neighbouring::NeighbouringDataManager *neighbouringDataManager;
    const hemelb::net::IOCommunicator& ioComms;

  private:
    void Initialise();

    void SetupReporting(); // set up the reporting file
    unsigned int OutputPeriod(unsigned int frequency);
    void HandleActors();
    void OnUnstableSimulation();
    void WriteLocalImages();
    void GenerateNetworkImages();
    /**
     * Updates the property caches record of which properties need to be calculated
     * and cached on this iteration.
     */
    void RecalculatePropertyRequirements();

    /**
     * Helper method to log simulation parameters related to stability and accuracy
     */
    void LogStabilityReport();

    hemelb::configuration::SimConfig *simConfig;
    hemelb::io::PathManager* fileManager;
    hemelb::reporting::Timers timings;
    hemelb::reporting::Reporter* reporter;
    hemelb::reporting::BuildInfo build_info;
    typedef std::multimap<unsigned long, unsigned long> MapType;

    MapType writtenImagesCompleted;
    MapType networkImagesCompleted;

    hemelb::lb::SimulationState* simulationState;

    /** Struct containing the configuration of various checkers/testers */
    const hemelb::configuration::SimConfig::MonitoringConfig* monitoringConfig;
    hemelb::lb::StabilityTester<latticeType>* stabilityTester;
    hemelb::lb::EntropyTester<latticeType>* entropyTester;
    /** Actor in charge of checking the maximum density difference across the domain */
    hemelb::lb::IncompressibilityChecker<hemelb::net::PhasedBroadcastRegular<> >* incompressibilityChecker;

    hemelb::colloids::ColloidController* colloidController;
    hemelb::net::Net communicationNet;

    const hemelb::util::UnitConverter* unitConverter;

    hemelb::extraction::IterableDataSource* propertyDataSource;
    hemelb::extraction::PropertyActor* propertyExtractor;

    hemelb::net::phased::StepManager* stepManager;
    hemelb::net::phased::NetConcern* netConcern;

    unsigned int imagesPerSimulation;
    unsigned int imagesPeriod;
    static const hemelb::LatticeTimeStep FORCE_FLUSH_PERIOD=1000;
};

#endif /* HEMELB_SIMULATIONMASTER_H */
