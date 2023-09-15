#pragma once
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdint>

typedef double distribn_t;
typedef int64_t site_t;
typedef unsigned Direction;
#define local_iolets_MaxSIZE 90

namespace hemelb {
 __global__ void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs(  distribn_t* GMem_dbl_fOld_b,
                                                                distribn_t* GMem_dbl_fNew_b,
                                                                distribn_t* GMem_dbl_MacroVars,
                                                                int64_t* GMem_int64_Neigh,
                                                                uint32_t* GMem_uint32_Wall_Link,
                                                                uint32_t* GMem_uint32_Iolet_Link,
                                                                uint64_t nArr_dbl,
                                                                distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom,
                                                                uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, bool write_GlobalMem);

    // GPU constant memory
     extern __constant__ site_t _Iolets_Inlet_Edge[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_InletWall_Edge[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_Inlet_Inner[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_InletWall_Inner[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_Outlet_Edge[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_OutletWall_Edge[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_Outlet_Inner[local_iolets_MaxSIZE];
     extern __constant__ site_t _Iolets_OutletWall_Inner[local_iolets_MaxSIZE];


    extern __constant__ unsigned int _NUMVECTORS;
    extern __constant__ double dev_tau;
    extern __constant__ double dev_minusInvTau;
    extern __constant__ double _Cs2;

    extern __constant__ bool _useWeightsFromFile;


    extern __constant__ int _InvDirections_19[19];

    extern __constant__ double _EQMWEIGHTS_19[19];

    extern __constant__ int _CX_19[19];
    extern __constant__ int _CY_19[19];
    extern __constant__ int _CZ_19[19];

    extern __constant__ int _WriteStep;
    extern __constant__ int _Send_MacroVars_DtH; // Writing MacroVariables to GPU global memory (Sending MacroVariables calculated during the collision-streaming kernels to the GPU Global mem).


__device__ __forceinline__  void _determine_Iolet_ID(int num_local_Iolets, site_t* iolets_ID_range, site_t fluid_Ind, int* IdInlet)
{
    // Loop over the number of local iolets (num_local_Iolets) and determine whether the fluid ID (fluid_Ind) falls whithin the range
    for (int i_local_iolet = 0; i_local_iolet<num_local_Iolets; i_local_iolet++)
    {
        // iolet range: [lower_limit,upper_limit)
        int64_t lower_limit = iolets_ID_range[3*i_local_iolet+1];   // Included in the fluids range
        int64_t upper_limit = iolets_ID_range[3*i_local_iolet+2];   // Value included in the fluids' range - CHANGED TO INCLUDE THE VALUE

        //if ((fluid_Ind - upper_limit +1) * (fluid_Ind - lower_limit) <= 0){       //When the upper_limit is NOT included
        if ((fluid_Ind - upper_limit) * (fluid_Ind - lower_limit) <= 0){                // When the upper_limit is included
            *IdInlet =(int)(iolets_ID_range[3*i_local_iolet]);
            return;
        }
    }// closes the loop over the local iolets
}
}; 
