
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#include "extraction/PropertyActor.h"

/*
// Added 18 July 2020 - Multithreading
#include "extraction/asynch_write.h"
//Threads Worker;
*/

namespace hemelb
{
  namespace extraction
  {
    PropertyActor::PropertyActor(const lb::SimulationState& simulationState,
                                 const std::vector<PropertyOutputFile*>& propertyOutputs,
                                 IterableDataSource& dataSource,
                                 reporting::Timers& timers,
                                 const net::IOCommunicator& ioComms) :
        simulationState(simulationState), timers(timers)
    {
      propertyWriter = new PropertyWriter(dataSource, propertyOutputs, ioComms);

      // Added 5 August 2020
      int max_sim_time = GetMaxSimTime();
    }

    PropertyActor::~PropertyActor()
    {
      delete propertyWriter;
    }

    int PropertyActor::GetMaxSimTime(){
      return(simulationState.GetTotalTimeSteps());
    }

    void PropertyActor::SetRequiredProperties(lb::MacroscopicPropertyCache& propertyCache)
    {
      const std::vector<LocalPropertyOutput*>& propertyOutputs = propertyWriter->GetPropertyOutputs();

      // Iterate over each property output spec.
      for (unsigned output = 0; output < propertyOutputs.size(); ++output)
      {
        const LocalPropertyOutput* propertyOutput = propertyOutputs[output];

        // Only consider the ones that are being written this iteration.
        if (propertyOutput->ShouldWrite(simulationState.GetTimeStep()))
        {
          const PropertyOutputFile* outputFile = propertyOutput->GetOutputSpec();

          // Iterate over each field.
          for (unsigned outputField = 0; outputField < outputFile->fields.size(); ++outputField)
          {
            // Set the cache to calculate each required field.
            switch (outputFile->fields[outputField].type)
            {
              case (OutputField::Pressure):
                propertyCache.densityCache.SetRefreshFlag();
                break;
              case OutputField::Velocity:
                propertyCache.velocityCache.SetRefreshFlag();
                break;
              case OutputField::ShearStress:
                propertyCache.wallShearStressMagnitudeCache.SetRefreshFlag();
                break;
              case OutputField::VonMisesStress:
                propertyCache.vonMisesStressCache.SetRefreshFlag();
                break;
              case OutputField::ShearRate:
                propertyCache.shearRateCache.SetRefreshFlag();
                break;
              case OutputField::StressTensor:
                propertyCache.stressTensorCache.SetRefreshFlag();
                break;
              case OutputField::Traction:
                propertyCache.tractionCache.SetRefreshFlag();
                break;
              case OutputField::TangentialProjectionTraction:
                propertyCache.tangentialProjectionTractionCache.SetRefreshFlag();
                break;
              case OutputField::MpiRank:
                // We don't actually have to cache anything to get the rank.
                break;
              default:
                // This assert should never trip. It only occurs when someone adds a new field to OutputField
                // and forgets adding a new case to the switch
                assert(false);
            }
          }
        }
      }
    }

/*
    int ThreadWork_Save_Files(Threads::Thread* thread){
      // propertyWriter->Write(simulationState.GetTimeStep());
      printf("Thread Id = %i \n", thread->Id);
      printf("Finished (%i)\n", thread->Id);
      return 0;
    }

    int PropertyActor::thread_Write()
    {
      propertyWriter->Write(simulationState.GetTimeStep());
      //std::thread thread_ForWrite(&thread_function);   // t starts running
      return 0;
    }
*/

    void PropertyActor::EndIteration()
    {
      timers[reporting::Timers::extractionWriting].Start();

      propertyWriter->Write(simulationState.GetTimeStep(), simulationState.GetTotalTimeSteps() );

      // Worker.WaitFinish();		//Wait for all threads to finish work
      // thread_Write();

      /**
        Initialise the thread (threadWrite) with the public member function (thread_Write) of the class PropertyActor
        and pass an object of the class (this), which defines this member function

      std::thread threadWrite(&PropertyActor::thread_Write, this);
      */

      //Worker.RunThreadsAsync(1, (PropertyActor::thread_Write()));

      //Worker.RunThreadsAsync(1, ThreadWork_Save_Files);
      //Worker.RunThreadsAsync(1, propertyWriter->Write(simulationState.GetTimeStep()) );

      timers[reporting::Timers::extractionWriting].Stop();
    }

  }
}
