// cuda_params.h
#ifndef cuda_params_h
#define cuda_params_h

#include <stdint.h> // to use uint64_t below
#include "units.h"

#include<hip/hip_runtime.h>
#include<hip/hip_runtime_api.h>

#define local_iolets_MaxSIZE 90 // This is the max array size with the iolet info (Iolet ID and fluid sites range, min and max, i.e. size = 3*local number of iolets). Assume that maximum number of iolets per RANK = local_iolets_MaxSIZE/3, i.e 30 here
																// Note the distinction between n_unique_local_Iolets and local iolets.
namespace hemelb
{

	__constant__ site_t _Iolets_Inlet_Edge[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_InletWall_Edge[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_Inlet_Inner[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_InletWall_Inner[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_Outlet_Edge[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_OutletWall_Edge[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_Outlet_Inner[local_iolets_MaxSIZE];
	__constant__ site_t _Iolets_OutletWall_Inner[local_iolets_MaxSIZE];


	// Struct to hold the info for the Iolets: Iolet ID and fluid sites ranges
	// Definition of the struct needs to be visible to all files
	// TODO: 	Need to switch array Iolets_ID_range
	// 				to be of type Flexible Array Member(FAM), which is of variable length
	struct Iolets{
		int n_local_iolets;						// 	Number of local Rank Iolets - NOTE: Some Iolet IDs may repeat, depending on the fluid ID numbering - see the value of unique iolets, (for example n_unique_LocalInlets_mInlet_Edge)
		site_t Iolets_ID_range[local_iolets_MaxSIZE]; 	//	Iolet ID and fluid sites range: [min_Fluid_Index, max_Fluid_Index], i.e 3 site_t values per iolet 
	};
	extern struct Iolets Inlet_Edge, Inlet_Inner, InletWall_Edge, InletWall_Inner;
	extern struct Iolets Outlet_Edge, Outlet_Inner, OutletWall_Edge, OutletWall_Inner;

	__constant__ unsigned int _NUMVECTORS;
	__constant__ double dev_tau;
	__constant__ double dev_minusInvTau;
	__constant__ int _InvDirections_19[19];
	__constant__ double _EQMWEIGHTS_19[19];
	__constant__ int _CX_19[19];
	__constant__ int _CY_19[19];
	__constant__ int _CZ_19[19];
	__constant__ double _Cs2;
	
	//============= (Paul)
	void d95901_set_numvectors(const unsigned int numvectors, hipError_t* status);
	
	void d95901_set_EQMWEIGHTS_19(const double* eqmweights, size_t size, hipError_t* status);
	
	void d95901_set_InvDirections_19(const hemelb::Direction* invdir, size_t size, hipError_t* status);
	
	void d95901_set_CX_19(const int* cx, size_t size, hipError_t* status);
	
	void d95901_set_CY_19(const int* cy, size_t size, hipError_t* status);
	
	void d95901_set_CZ_19(const int* cz, size_t size, hipError_t* status);
	
	void d95901_set_dev_tau(const double* devtau, size_t size, hipError_t* status);
	
	void d95901_set_dev_minusInvTau(const double* devminusInvTau, size_t size, hipError_t* status);
	
	void d95901_set_Cs2(const double* cs2, size_t size, hipError_t* status);
	
	//==============
	

	__constant__ int _WriteStep=1000;
	__constant__ int _Send_MacroVars_DtH=1000;


	inline void check_cuda_errors(const char *filename, const int line_number, int myProc);

	// Declare global cuda functions here - Callable from within a class
	// __global__ void GPU_Collide_testing(long lower_limit, long upper_limit);

	__global__ void GPU_CalcMacroVars_Swap(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, unsigned int nArr_dbl, long long lower_limit, long long upper_limit, int time_Step);

	__global__ void GPU_CalcMacroVars(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_MacroVars, unsigned int nArr_dbl, long long lower_limit, long long upper_limit);

	__global__ void GPU_CollideStream_1_PreSend(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, int64_t* GMem_int64_Neigh, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);

	__global__ void GPU_CollideStream_1_PreReceive(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);

	__global__ void GPU_CollideStream_1_PreReceive_SaveMacroVars(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step);
	__global__ void GPU_CollideStream_1_PreReceive_noSave(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);
	__global__ void GPU_CollideStream_1_PreReceive_new(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);

	__global__ void GPU_CollideStream_mWallCollision_sBB(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);
	__global__ void GPU_CollideStream_mWallCollision_sBB_PreRec(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step);

	__global__ void GPU_CollideStream_3_NashZerothOrderPressure(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);
	__global__ void GPU_CollideStream_3_NashZerothOrderPressure_new(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, site_t* iolets_ID_range);

	__global__ void GPU_CollideStream_3_NashZerothOrderPressure_Inlet_Inner(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_3_NashZerothOrderPressure_Inlet_Edge(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_3_NashZerothOrderPressure_Outlet_Inner(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_3_NashZerothOrderPressure_Outlet_Edge(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);

	__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, Iolets Iolets_info);

//	Kernels for Velocity & Pressure BCs currently in use:
	// Pressure BCs (NASHZEROTHORDERPRESSUREIOLET):
	__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure_v2(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, site_t* GMem_Iolets_info);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_v2( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, site_t* GMem_Iolets_info);
	// Velocity BCs (LADDIOLET)
	__global__ void GPU_CollideStream_Iolets_Ladd_VelBCs(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, uint64_t nArr_dbl, distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step);
//

	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_new( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, site_t* iolets_ID_range);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, Iolets Iolets_info);

	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Inlet_Inner( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Inlet_Edge( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Outlet_Inner( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Outlet_Edge( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);

	__global__ void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs(	distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, uint64_t nArr_dbl, distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step);

	__global__ void GPU_SwapOldAndNew(distribn_t* __restrict__ GMem_dbl_fOld_b, distribn_t* __restrict__ GMem_dbl_fNew_b, site_t nArr_dbl, site_t lower_limit, site_t upper_limit);

	__global__ void GPU_StreamReceivedDistr(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, site_t* GMem_int64_streamInd, site_t nArr_dbl, site_t upper_limit);
	__global__ void GPU_StreamReceivedDistr_fOldTofOld(distribn_t* GMem_dbl_fOld_b, site_t* GMem_int64_streamInd, site_t nArr_dbl, site_t upper_limit);

	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB(distribn_t* GMem_dbl_fOld_b,
										distribn_t* GMem_dbl_fNew_b,
										distribn_t* GMem_dbl_MacroVars,
										site_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										site_t nArr_dbl,
										site_t lower_limit_MidFluid, site_t upper_limit_MidFluid,
										site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, int time_Step);



//==============================================================================
/**
	Device function to investigate which Iolet Ind corresponds to a fluid with index fluid_Ind
 		To be used for the inlet/outlet related collision-streaming kernels
 		Checks through the local iolets (inlet/outlet) to determine the correct iolet ID
 			each iolet has fluid sites with indices in the range: [lower_limit,upper_limit]
 	Function returns the iolet ID value: IdInlet.
*/
__device__ __forceinline__ void _determine_Iolet_ID(int num_local_Iolets, site_t* iolets_ID_range, site_t fluid_Ind, int* IdInlet)
{
	// Loop over the number of local iolets (num_local_Iolets) and determine whether the fluid ID (fluid_Ind) falls whithin the range
	for (int i_local_iolet = 0; i_local_iolet<num_local_Iolets; i_local_iolet++)
	{
		// iolet range: [lower_limit,upper_limit)
		int64_t lower_limit = iolets_ID_range[3*i_local_iolet+1];	// Included in the fluids range
		int64_t upper_limit = iolets_ID_range[3*i_local_iolet+2];	// Value included in the fluids' range - CHANGED TO INCLUDE THE VALUE

		//if ((fluid_Ind - upper_limit +1) * (fluid_Ind - lower_limit) <= 0){	 	//When the upper_limit is NOT included
		if ((fluid_Ind - upper_limit) * (fluid_Ind - lower_limit) <= 0){ 				// When the upper_limit is included
			*IdInlet = (int)iolets_ID_range[3*i_local_iolet];
			return;
		}
	}// closes the loop over the local iolets
}
//==============================================================================


}
#endif
