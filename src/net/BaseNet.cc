
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

/*! \file net.cc
 \brief In this file the functions useful to discover the topology used and
 to create and delete the domain decomposition and the various
 buffers are defined.
 */

#include <cstdlib>
#include <cmath>
#include <cstdio>

#include "net/BaseNet.h"
#include "util/utilityFunctions.h"
#include "util/Vector3D.h"
#include "net/IOCommunicator.h"


namespace hemelb
{
  namespace net
  {

    void BaseNet::Dispatch()
    {
      Send();
      Receive();
      Wait();
    }

    BaseNet::BaseNet(const MpiCommunicator &commObject) :
        BytesSent(0), SyncPointsCounted(0), communicator(commObject)
    {
    }

    void BaseNet::Receive()
    {
      ReceiveGathers();
      ReceiveGatherVs();
      ReceiveAllToAll();
      // Ensure collectives are called before point-to-point, as some implementing mixins implement collectives via point-to-point
      ReceivePointToPoint();
    }

    void BaseNet::Send()
    {
      SendGathers();
      SendGatherVs();
      SendAllToAll();
      // Ensure collectives are called before point-to-point, as some implementing mixins implement collectives via point-to-point
      SendPointToPoint();
    }

//=========================================================================================================
#ifdef HEMELB_USE_GPU
    bool BaseNet::Synchronise_memCpy_GPU_CPU_domainEdge()
    {
      int myPiD = communicator.Rank();
      if (myPiD!=0) {
        cudaStreamSynchronize(stream_memCpy_GPU_CPU_domainEdge_new2);
        //printf("Synchronisation point for cuda stream from BaseNet and rank # %d \n\n", myPiD);
      }
      return true;
    }

    // Create the cuda stream
    bool BaseNet::Create_stream_memCpy_GPU_CPU_domainEdge_new2()
    {
      int myPiD = communicator.Rank();
      if (myPiD!=0) {
        cudaStreamCreate(&stream_memCpy_GPU_CPU_domainEdge_new2);
        //printf("Created cuda stream from BaseNet and rank # %d \n\n", myPiD);
      }
      return true;
    }

    // Destroy the cuda stream
    bool BaseNet::Destroy_stream_memCpy_GPU_CPU_domainEdge_new2()
    {
      int myPiD = communicator.Rank();
      if (myPiD!=0) {
        cudaStreamDestroy(stream_memCpy_GPU_CPU_domainEdge_new2);
        //printf("Destroyed cuda stream from BaseNet and rank # %d \n\n", myPiD);
      }
      return true;
    }

    // Get the cuda stream - private member
    cudaStream_t BaseNet::Get_stream_memCpy_GPU_CPU_domainEdge_new2()
    {
      return stream_memCpy_GPU_CPU_domainEdge_new2;
    }
#endif
//=========================================================================================================


    void BaseNet::Wait()
    {
      SyncPointsCounted++; //DTMP: counter for monitoring purposes.

      WaitGathers();
      WaitGatherVs();
      WaitPointToPoint();
      WaitAllToAll();

      displacementsBuffer.clear();
      countsBuffer.clear();
    }

    std::vector<int> & BaseNet::GetDisplacementsBuffer()
    {
      displacementsBuffer.push_back(std::vector<int>());
      return displacementsBuffer.back();
    }

    std::vector<int> & BaseNet::GetCountsBuffer()
    {
      countsBuffer.push_back(std::vector<int>());
      return countsBuffer.back();
    }

  }
}
