// cuda_params.h
#ifndef cuda_params_h
#define cuda_params_h

#include <stdint.h> // to use uint64_t below
#include "units.h"

#define local_iolets_MaxSIZE 90 // This is the max array size with the iolet info (Iolet ID and fluid sites range, min and max, i.e. size = 3*local number of iolets). Assume that maximum number of iolets per RANK = local_iolets_MaxSIZE/3, i.e 30 here
																// Note the distinction between n_unique_local_Iolets and local iolets.

#define frequency_WriteGlobalMem 100 // Frequency to write macroVariables to GPU global memory

namespace hemelb
{

	extern __constant__ site_t _Iolets_Inlet_Edge[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_InletWall_Edge[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_Inlet_Inner[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_InletWall_Inner[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_Outlet_Edge[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_OutletWall_Edge[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_Outlet_Inner[local_iolets_MaxSIZE];
	extern __constant__ site_t _Iolets_OutletWall_Inner[local_iolets_MaxSIZE];


	// Struct to hold the info for the Iolets: Iolet ID and fluid sites ranges
	// Definition of the struct needs to be visible to all files
	struct Iolets{
		int n_local_iolets;						// 	Number of local Rank Iolets - NOTE: Some Iolet IDs may repeat, depending on the fluid ID numbering - see the value of unique iolets, (for example n_unique_LocalInlets_mInlet_Edge)
		site_t Iolets_ID_range[local_iolets_MaxSIZE]; 	//	Iolet ID and fluid sites range: [min_Fluid_Index, max_Fluid_Index], i.e 3 site_t values per iolet
	};
	extern struct Iolets Inlet_Edge, Inlet_Inner, InletWall_Edge, InletWall_Inner;
	extern struct Iolets Outlet_Edge, Outlet_Inner, OutletWall_Edge, OutletWall_Inner;

	extern __constant__ unsigned int _NUMVECTORS;
	extern __constant__ double dev_tau;
	extern __constant__ double dev_minusInvTau;
	extern __constant__ int _InvDirections_19[19];
	extern __device__ __constant__ double _EQMWEIGHTS_19[19];
	extern __constant__ int _CX_19[19];
	extern __constant__ int _CY_19[19];
	extern __constant__ int _CZ_19[19];
	extern __constant__ double _Cs2;
	extern __constant__ bool _useWeightsFromFile;

	extern __constant__ int _WriteStep; // Not used

	// Variable for saving MacroVariables to GPU global memory in each of the collision-streaming kernels
	// Then Function Read_Macrovariables_GPU_to_CPU in void LBM<LatticeType>::EndIteration() will do the DtH mem.copy
	extern __constant__ int _Send_MacroVars_DtH;



	inline void check_cuda_errors(const char *filename, const int line_number, int myProc);

	// Declare global cuda functions here - Callable from within a class
	// __global__ void GPU_Collide_testing(long lower_limit, long upper_limit);

	//============================================================================
	//
	// Currently using the following GPU kernels:
	__global__ void GPU_Check_Stability(distribn_t* GMem_dbl_fOld_b,
																											distribn_t* GMem_dbl_fNew_b,
																											int* d_Stability_flag,
																											site_t nArr_dbl,
																											site_t lower_limit, site_t upper_limit,
																											int time_Step);

	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB(distribn_t* GMem_dbl_fOld_b,
										distribn_t* GMem_dbl_fNew_b,
										distribn_t* GMem_dbl_MacroVars,
										site_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										site_t nArr_dbl,
										site_t lower_limit_MidFluid, site_t upper_limit_MidFluid,
										site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, bool write_GlobalMem);

	//	Kernels for Velocity & Pressure BCs:
	// Pressure BCs (NASHZEROTHORDERPRESSUREIOLET):
	__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure_v2(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																																			int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity,
																																			float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit,
																																			uint64_t totalSharedFs, bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info);

	__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																																		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity,
																																		float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit,
																																		uint64_t totalSharedFs, bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info);

	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																													int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
																													float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																													bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info);


  __global__ void GPU_CollideStream_wall_sBB_iolet_Nash_v2( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																														int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
																														float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																														bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info);

	// Velocity BCs (LADDIOLET)
	__global__ void GPU_CollideStream_Iolets_Ladd_VelBCs(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																												int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, uint64_t nArr_dbl,
																												distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
																												uint64_t totalSharedFs, bool write_GlobalMem);

	__global__ void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs(	distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																																	int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link,
																																	uint64_t nArr_dbl, distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
																																	uint64_t totalSharedFs, bool write_GlobalMem);

// Related to the wall momentum correction terms evaluation on the GPU
	__global__ void GPU_WallMom_correction_File_prefactor(distribn_t* GMem_dbl_wallMom_prefactor_correction,
																										distribn_t *GMem_dbl_WallMom,
																										uint32_t* GMem_uint32_Iolet_Link,
																										int num_local_Iolets, site_t* GMem_Iolets_info,
																										distribn_t* GMem_Inlet_velocityTable,
																										site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
																										site_t lower_limit, site_t upper_limit, unsigned long time_Step, unsigned long total_TimeSteps);

  __global__ void GPU_WallMom_correction_File_prefactor_v2(
																										distribn_t* GMem_dbl_wallMom_prefactor_correction,
																										distribn_t* GMem_dbl_WallMom,
																										uint32_t* GMem_uint32_Iolet_Link,
																										int num_local_Iolets, Iolets Iolets_info,
																										distribn_t* GMem_Inlet_velocityTable,
																										site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
																										site_t lower_limit, site_t upper_limit,
																										unsigned long time_Step, unsigned long total_TimeSteps);

  __global__ void GPU_WallMom_correction_File_prefactor_NoIoletIDSearch(
																		distribn_t* GMem_dbl_wallMom_prefactor_correction,
																		distribn_t* GMem_dbl_WallMom,
																		uint32_t* GMem_uint32_Iolet_Link,
																		int IdInlet,
																		distribn_t* GMem_Inlet_velocityTable,
																		site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
																		site_t lower_limit, site_t upper_limit,
																		unsigned long time_Step, unsigned long total_TimeSteps);

//============================================================================
// Testing:
__global__ void GPU_Check_Coordinates(int64_t *GMem_Coords_iolets,
																site_t start_Fluid_ID_givenColStreamType,
																site_t lower_limit, site_t upper_limit
															);
	__global__ void GPU_Check_Velocity_BCs_table_weights(	int64_t **GMem_pp_int_weightsTable_coord,
																												distribn_t **GMem_pp_dbl_weightsTable_wei,
																												int inlet_ID,
																												distribn_t* GMem_Inlet_velocityTable,
																												int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets);

	/*__global__ void GPU_Check_Velocity_BCs_table_weights(int **GMem_pp_int_weightsTable_coord, int inlet_ID,
																											distribn_t* GMem_Inlet_velocityTable,
																											int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets);*/

  __global__ void GPU_Check_Velocity_BCs_table_weights_directArr(	int *GMem_p_int_weightsTable_coord, int inlet_ID,
																																	distribn_t* GMem_Inlet_velocityTable,
																																	int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets
																																);

  __global__ void GPU_Check_Velocity_BCs_table_weights_directArr_v2(int *GMem_p_int_weightsTable_coord_x,
																																		int *GMem_p_int_weightsTable_coord_y,
																																		int *GMem_p_int_weightsTable_coord_z,
																																		int inlet_ID,
																																		distribn_t* GMem_Inlet_velocityTable,
																																		int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets);
	//
	//============================================================================

	__global__ void GPU_WallMom_correction_File_Weights(int64_t *GMem_Coords_iolets, int64_t **GMem_pp_int_weightsTable_coord,
																											distribn_t **GMem_pp_dbl_weightsTable_wei, int64_t* GMem_index_key_weightTable,
																											distribn_t *GMem_dbl_WallMom, float* GMem_ioletNormal,
																											uint32_t* GMem_uint32_Iolet_Link,
																											int inlet_ID,
																											distribn_t* GMem_Inlet_velocityTable,
																											int n_arr_elementsInCurrentInlet_weightsTable,
																											site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
																											site_t lower_limit, site_t upper_limit, unsigned long time_Step, unsigned long total_TimeSteps);

	__global__ void GPU_WallMom_correction_File_Weights_NoSearch(int64_t *GMem_Coords_iolets, int64_t **GMem_pp_int_weightsTable_coord,
																											distribn_t **GMem_pp_dbl_weightsTable_wei, int64_t* GMem_index_key_weightTable,
																											distribn_t* GMem_weightTable,
																											distribn_t *GMem_dbl_WallMom, float* GMem_ioletNormal,
																											uint32_t* GMem_uint32_Iolet_Link,
																											int inlet_ID,
																											distribn_t* GMem_Inlet_velocityTable,
																											int n_arr_elementsInCurrentInlet_weightsTable,
																											site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
																											site_t lower_limit, site_t upper_limit, unsigned long time_Step, unsigned long total_TimeSteps);


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



	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_new( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, site_t* iolets_ID_range);


	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Inlet_Inner( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Inlet_Edge( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Outlet_Inner( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Outlet_Edge( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars, int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity, float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets);


	__global__ void GPU_SwapOldAndNew(distribn_t* __restrict__ GMem_dbl_fOld_b, distribn_t* __restrict__ GMem_dbl_fNew_b, site_t nArr_dbl, site_t lower_limit, site_t upper_limit);

	__global__ void GPU_StreamReceivedDistr(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, site_t* GMem_int64_streamInd, site_t nArr_dbl, site_t upper_limit);
	__global__ void GPU_StreamReceivedDistr_fOldTofOld(distribn_t* GMem_dbl_fOld_b, site_t* GMem_int64_streamInd, site_t nArr_dbl, site_t upper_limit);



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

	/* Device function to evaluate second moment of a distr. function
	/* Despite its name, this method does not compute the whole pi tensor (i.e. momentum flux tensor). What it does is
							 * computing the second moment of a distribution function. If this distribution happens to be f_eq, the resulting
							 * tensor will be the equilibrium part of pi. However, if the distribution function is f_neq, the result WON'T be
							 * the non equilibrium part of pi. In order to get it, you will have to multiply by (1 - timestep/2*tau)
							 *
							 * @param f distribution function
							 * @return second moment of the distribution function f
	*/
	__device__ __forceinline__ double *_CalculatePiTensor(const distribn_t* const f)
	{
			static double ret_SecMomDistrFunc[6];

			/*
			// Fill in (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
			for (int ii = 0; ii < 3; ++ii)
			{
				for (int jj = 0; jj <= ii; ++jj)
				{
					ret[ii][jj] = 0.0;
						for (unsigned int l = 0; l < DmQn::NUMVECTORS; ++l)
						{
							ret[ii][jj] += f[l] * DmQn::discreteVelocityVectors[ii][l]
													* DmQn::discreteVelocityVectors[jj][l];
						}
				}
			}
			*/

			// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
			// and saves these with this order in the array ret_SecMomDistrFunc

			// Element (0,0)
			ret_SecMomDistrFunc[0] = 0.0;
			for (unsigned int l = 0; l < _NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[0] += f[l] * _CX_19[l]* _CX_19[l];
			}

			// Element (1,0)
			ret_SecMomDistrFunc[1] = 0.0;
			for (unsigned int l = 0; l < _NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[1] += f[l] * _CY_19[l]* _CX_19[l];
			}

			// Element (1,1)
			ret_SecMomDistrFunc[2] = 0.0;
			for (unsigned int l = 0; l < _NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[2] += f[l] * _CY_19[l]* _CY_19[l];
			}

			// Element (2,0)
			ret_SecMomDistrFunc[3] = 0.0;
			for (unsigned int l = 0; l < _NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[3] += f[l] * _CZ_19[l]* _CX_19[l];
			}

			// Element (2,1)
			ret_SecMomDistrFunc[4] = 0.0;
			for (unsigned int l = 0; l < _NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[4] += f[l] * _CZ_19[l]* _CY_19[l];
			}

			// Element (2,2)
			ret_SecMomDistrFunc[5] = 0.0;
			for (unsigned int l = 0; l < _NUMVECTORS; ++l)
			{
				ret_SecMomDistrFunc[5] += f[l] * _CZ_19[l]* _CZ_19[l];
			}


			return ret_SecMomDistrFunc;
	}


}
#endif
