// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "lb/InitialCondition.h"
#include <boost/optional/optional_io.hpp>

namespace hemelb {
  namespace lb {

    // InitialConditionBase

    InitialConditionBase::InitialConditionBase() {
    }
    
    InitialConditionBase::InitialConditionBase(boost::optional<LatticeTimeStep> t) : initial_time(t) {
      // IZ Feb 2024 - Debugging (checkpointing) - Remove later
      // It does enter here - Brings over the correct restarting time as indicated in the input file!!!
      //    <initial_time units="lattice" value="5000" />
      //std::cout << "ENTERS InitialConditionBase (2)!!! with initial time = " << t << std::endl;
      //std::cout << "ENTERS InitialConditionBase (3)!!! with initial time = " << initial_time << std::endl;
    }


    void InitialConditionBase::SetTime(SimulationState* sim) const {
      std::cout << "Entering SetTime (InitialConditionBase)!!!!!!!!! Initial time: " << initial_time << std::endl;
      //printf("Entering SetTime (InitialConditionBase)!!!!!!!!! Initial time: %d \n\n", initial_time);
      if (initial_time){
        sim->timeStep = *initial_time;

        // IZ 2024 - Debugging - Remove later
        uint64_t time_currentStep = sim->GetTimeStep();
  			printf("Current Time-Step as set in InitialConditionBase::SetTime %ld \n", time_currentStep);
        //
      }
    }


    // Equilibrium

    EquilibriumInitialCondition::EquilibriumInitialCondition() :
      InitialConditionBase(),
      density(1.0), mom_x(0.0), mom_y(0.0), mom_z(0.0) {
    }

    EquilibriumInitialCondition::EquilibriumInitialCondition(
      boost::optional<LatticeTimeStep> t0,
      distribn_t rho,
      distribn_t mx, distribn_t my, distribn_t mz) :
      InitialConditionBase(t0),
      density(rho),
      mom_x(mx), mom_y(my), mom_z(mz) {
    }

    CheckpointInitialCondition::CheckpointInitialCondition(boost::optional<LatticeTimeStep> t0, const std::string& cp)
      : InitialConditionBase(t0), cpFile(cp) {
    }

    // InitialCondition - sum type container

    // Visitor for setting time
    struct TSetter {
      using result_type = void;
      template <typename T>
      void operator()(T& t) const {
        t.SetTime(ss);
      }
      SimulationState* ss;
    };


    
    void InitialCondition::SetTime(SimulationState* sim) const {
      const ICVar* self = this;
      boost::apply_visitor(TSetter{sim}, *self);
    }
    

    // IZ - Jan 2024
    /*
    void CheckpointInitialCondition::SetTime(SimulationState* sim) const {

      //sim->timeStep = 0;
      //auto distributionInputPtr = std::make_unique<extraction::LocalDistributionInput>(sim);
      //distributionInputPtr = extraction::LocalDistributionInput->Get_restart_Time_Checkpointing();
      //auto time_restart = distributionInputPtr->Get_restart_Time_Checkpointing();

    //    //auto distributionInputPtr = std::make_unique<extraction::LocalDistributionInput>(sim);
    //  //distributionInputPtr->LoadDistribution(latDat, initial_time);
      //printf("Inside Initial Condition(.cc) (1) - initial_time = %d \n\n", time_restart);
    }
    */
    //

    // See InitialCondtions.hpp for setting Fs (distributions)

    // Visitor for factory function
    struct ICMaker {
      using result_type = InitialCondition;

      template <typename T>
      InitialCondition operator()(T) const {
        throw Exception() << "Trying to make an InitialCondition from unknown type of config";
      }

      InitialCondition operator()(const configuration::EquilibriumIC& cfg) const {
        auto rho = cfg.unitConverter->ConvertPressureToLatticeUnits(cfg.p_mmHg) / Cs2;
        return EquilibriumInitialCondition{cfg.t0, rho};
      }

      InitialCondition operator()(const configuration::CheckpointIC& cfg) const {
        return CheckpointInitialCondition{cfg.t0, cfg.cpFile};
      }
    };

    // Factory function just delegates to visitor
    InitialCondition InitialCondition::FromConfig(const configuration::ICConfig& conf) {
      return boost::apply_visitor(ICMaker{}, conf);
    }
  }
}
