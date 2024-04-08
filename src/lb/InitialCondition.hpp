// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_INITIALCONDITION_HPP
#define HEMELB_LB_INITIALCONDITION_HPP


namespace hemelb {
  namespace lb {
    template<class LatticeType>
    struct FSetter {
      using result_type = void;
      template <typename T>
      void operator()(T t) const {
        t.template SetFs<LatticeType>(latDat, ioComms, sim);
      }
      geometry::LatticeData* latDat;
      const net::IOCommunicator& ioComms;
      SimulationState* sim;
    };


    template<class LatticeType>
    void InitialCondition::SetFs(geometry::LatticeData* latDat, const net::IOCommunicator& ioComms, SimulationState* sim) const {
      boost::apply_visitor(FSetter<LatticeType>{latDat, ioComms, sim}, *this);
    }

    template<class LatticeType>
    void EquilibriumInitialCondition::SetFs(geometry::LatticeData* latDat, const net::IOCommunicator& ioComms, SimulationState* sim) const {

      printf("Setting Distr. Funstions - Case EquilibriumInitialCondition \n\n");

      distribn_t f_eq[LatticeType::NUMVECTORS];
      LatticeType::CalculateFeq(density, mom_x, mom_y, mom_z, f_eq);

      for (site_t i = 0; i < latDat->GetLocalFluidSiteCount(); i++) {
        distribn_t* f_old_p = this->GetFOld(latDat, i * LatticeType::NUMVECTORS);
        distribn_t* f_new_p = this->GetFNew(latDat, i * LatticeType::NUMVECTORS);

        for (unsigned int l = 0; l < LatticeType::NUMVECTORS; l++) {
          f_new_p[l] = f_old_p[l] = f_eq[l];
        }
      }

      // Set the initial time of the simulation
      // Assumption that t_start=1 when starting from an equilibrium configuration
      sim->SetTimeStep(1);
      sim->SetInitTimeStep(1);
    }


    template<class LatticeType>
    void CheckpointInitialCondition::SetFs(geometry::LatticeData* latDat, const net::IOCommunicator& ioComms, SimulationState* sim) const {

      //printf("Setting Distr. Funstions - Case CheckpointInitialCondition \n\n");

      auto distributionInputPtr = std::make_unique<extraction::LocalDistributionInput>(cpFile, ioComms);
      distributionInputPtr->LoadDistribution(latDat, initial_time);

      auto time_restart = distributionInputPtr->Get_restart_Time_Checkpointing();
      uint64_t time_to_pass = time_restart +1;

      sim->SetTimeStep(time_to_pass);
      sim->SetInitTimeStep(time_to_pass);

      printf("Inside Initial Condition(.hpp) (1a) - initial_time = %d, restart_time = %ld, time_passed_to_SimulationState = %ld \n\n", initial_time, time_restart, time_to_pass);
    }

    //--------------------------------
    // Testing Checkpointing functionality - Initial Time

    // Visitor for setting time
    struct InitTimeSetter {
      using result_type = void;
      template <typename T>
      void operator()(T t) const {
        t.SetInitTime(ss, ioComms);
      }
      SimulationState* ss;
      const net::IOCommunicator& ioComms;
    };

    inline void InitialCondition::SetInitTime(SimulationState* sim, const net::IOCommunicator& ioComms) const {
      boost::apply_visitor(InitTimeSetter{sim, ioComms}, *this);
    }

    inline void EquilibriumInitialCondition::SetInitTime(SimulationState* sim, const net::IOCommunicator& ioComms) const {

      printf(" - Setting Initial Time - Case EquilibriumInitialCondition \n\n");

    }

    inline void CheckpointInitialCondition::SetInitTime(SimulationState* sim, const net::IOCommunicator& ioComms) const {

      printf(" - Setting Initial Time - Case CheckpointInitialCondition \n\n");

      auto distributionInputPtr = std::make_unique<extraction::LocalDistributionInput>(cpFile, ioComms);
      auto time_restart = distributionInputPtr->Get_restart_Time_Checkpointing();
      printf("Inside Initial Condition(.hpp) (1) - initial restart_time = %ld \n\n", time_restart);

      uint64_t time_to_pass = time_restart +1;
      sim->SetTimeStep(time_to_pass);
    }

    //--------------------------------

  }
}

#endif
