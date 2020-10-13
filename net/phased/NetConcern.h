
// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_NET_PHASED_NETCONCERN_H
#define HEMELB_NET_PHASED_NETCONCERN_H

#include "net/phased/Concern.h"
#include "net/phased/steps.h"

//#include "lb/lb.h"

namespace hemelb
{
  namespace net
  {
    namespace phased
    {
      class NetConcern : public Concern
      {
        public:
          NetConcern(net::BaseNet & net) :
              net(net)
          {
          }
          /*
          hemelb::net::Net& mNet_cuda_stream = *mNet;	// Needs the constructor and be initialised
  				cudaStream_t Cuda_Stream_memCpy_GPU_CPU_domainEdge = mNet_cuda_stream.Get_stream_memCpy_GPU_CPU_domainEdge_new2();
  				cudaStreamCreate(&Cuda_Stream_memCpy_GPU_CPU_domainEdge);
  				*/
  				// Or this one: mNet_stream_cuda.Create_stream_memCpy_GPU_CPU_domainEdge_new2(); // Which one is correct? Does it actually create the stream and then it imposes a barrier in net::BaseNet::Send


          bool CallAction(int action)
          {
            switch (static_cast<phased::steps::Step>(action))
            {
              case phased::steps::Send:

#ifdef HEMELB_USE_GPU
#ifndef HEMELB_CUDA_AWARE_MPI
                // Synchronisation barrier - Barrier for stream created for the asynch. memcpy at domain edges
                // Only called if NO CUDA-aware mpi
                net.Synchronise_memCpy_GPU_CPU_domainEdge();
#endif
#endif
                net.Send();
                return true;
              case phased::steps::Receive:
                net.Receive();
                return true;
              case phased::steps::Wait:
                net.Wait();
                return true;

              default:
                return false;
            }
          }

        private:
          net::BaseNet & net;
      };
    }
  }
}

#endif
