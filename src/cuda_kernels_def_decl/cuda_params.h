// cuda_params.h
#ifndef cuda_params_h
#define cuda_params_h

#include <stdint.h> // to use uint64_t below
#include "units.h"
#include "cuda_kernels_def_decl/deviceAPI.h"

#define local_iolets_MaxSIZE 90 // This is the max array size with the iolet info (Iolet ID and fluid sites range, min and max, i.e. size = 3*local number of iolets). Assume that maximum number of iolets per RANK = local_iolets_MaxSIZE/3, i.e 30 here
																// Note the distinction between n_unique_local_Iolets and local iolets.

#define frequency_WriteGlobalMem 1000 // Frequency to write macroVariables to GPU global memory

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
	extern __constant__ double _iStressParameter;

	extern __constant__ int _WriteStep; // Not used

	// Variable for saving MacroVariables to GPU global memory in each of the collision-streaming kernels
	// Then Function Read_Macrovariables_GPU_to_CPU in void LBM<LatticeType>::EndIteration() will do the DtH mem.copy
	extern __constant__ int _Send_MacroVars_DtH;

	// Added for the Sponge Layer - LES implementation
	extern __constant__ double dev_smag_cnst;


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


	// Evaluate the wall shear stress magnitude
	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress(distribn_t* GMem_dbl_fOld_b,
										distribn_t* GMem_dbl_fNew_b,
										distribn_t* GMem_dbl_MacroVars,
										site_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										site_t nArr_dbl,
										site_t lower_limit_MidFluid, site_t upper_limit_MidFluid,
										site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, bool write_GlobalMem,
										distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
										unsigned long time_Step, int MPI_Rank);

  // Evaluate the wall shear stress magnitude
	//	& Sponge Layer - LES
	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress(
		distribn_t* GMem_dbl_fOld_b,
		distribn_t* GMem_dbl_fNew_b,
		distribn_t* GMem_dbl_MacroVars,
		site_t* GMem_int64_Neigh,
		uint32_t* GMem_uint32_Wall_Link,
		site_t nArr_dbl,
		site_t lower_limit_MidFluid, site_t upper_limit_MidFluid,
		site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, bool write_GlobalMem,
		distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
		unsigned long time_Step, int MPI_Rank,
		distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
	);


	//	Kernels for Velocity & Pressure BCs:
	// Pressure BCs (NASHZEROTHORDERPRESSUREIOLET):
	__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure_v2(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																																			int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity,
																																			float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit,
																																			uint64_t totalSharedFs, bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info);

  // Overloading -  Case of Sponge LAyer - LES implementation
	__global__
	void GPU_CollideStream_Iolets_NashZerothOrderPressure_v2(
		distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity,
		float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit,
		uint64_t totalSharedFs, bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info,
		unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
	);


	__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																																		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity,
																																		float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit,
																																		uint64_t totalSharedFs, bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info);

  // Overloading -  Case of Sponge LAyer - LES implementation
	__global__
	void GPU_CollideStream_Iolets_NashZerothOrderPressure(
		distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, double* GMem_ghostDensity,
		float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl,uint64_t lower_limit, uint64_t upper_limit,
		uint64_t totalSharedFs, bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info,
		unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
	);


	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																													int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
																													float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																													bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info);


  __global__ void GPU_CollideStream_wall_sBB_iolet_Nash_v2( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
																														int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
																														float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																														bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info);


__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
	 		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
			float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
			bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info,
			distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal);

// Overloading -  Case of Sponge LAyer - LES implementation
__global__
void GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress(
	distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
	int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
	float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
	bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info,
	distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
	unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
);

__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress( distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
			float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
			bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info,
			distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal);

// Overloading -  Case of Sponge LAyer - LES implementation
__global__
void GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress(
	distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
	int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link, distribn_t* GMem_ghostDensity,
	float* GMem_inletNormal, int nInlets, uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
	bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info,
	distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
	unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
);


//------------------------------------------------------------------------------
	// Velocity BCs (LADDIOLET)
__global__
	void GPU_CollideStream_Iolets_Ladd_VelBCs(
		distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b,
		distribn_t* GMem_dbl_MacroVars,
		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, uint64_t nArr_dbl,
		distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
		uint64_t totalSharedFs, bool write_GlobalMem);

// Overloading -  Case of Sponge LAyer - LES implementation
__global__
		void GPU_CollideStream_Iolets_Ladd_VelBCs(
			distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b,
			distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Iolet_Link, uint64_t nArr_dbl,
			distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
			uint64_t totalSharedFs, bool write_GlobalMem,
			unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
		);
//------------------------------------------------------------------------------


	__global__ void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs(	distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link,
			uint64_t nArr_dbl, distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
			uint64_t totalSharedFs, bool write_GlobalMem);

//==============================================================================
	__global__
	void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress(
		distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
		int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link,
		uint64_t nArr_dbl, distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
		uint64_t totalSharedFs, bool write_GlobalMem,
		distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal);

		__global__
		void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress(
			distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh, uint32_t* GMem_uint32_Wall_Link, uint32_t* GMem_uint32_Iolet_Link,
			uint64_t nArr_dbl, distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom, uint64_t lower_limit, uint64_t upper_limit,
			uint64_t totalSharedFs, bool write_GlobalMem,
			distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
			unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
		);

//==============================================================================



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
																		unsigned long time_Step, unsigned long total_TimeSteps, unsigned long start_time);

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
																											site_t lower_limit, site_t upper_limit, unsigned long time_Step, unsigned long total_TimeSteps, unsigned long start_time);


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



//==============================================================================
// May 2023
// Struct to contain the array for the second moment of distr. functions
// 6 elements are sufficient
struct structSecMomDistrFun
{
	//array declared inside structure
	double arr[6];
};

//==============================================================================
	/* Device function to evaluate second moment of a distr. function
	/* Despite its name, this method does not compute the whole pi tensor (i.e. momentum flux tensor). What it does is
	* computing the second moment of a distribution function. If this distribution happens to be f_eq, the resulting
	* tensor will be the equilibrium part of pi. However, if the distribution function is f_neq, the result WON'T be
	* the non equilibrium part of pi. In order to get it, you will have to multiply by (1 - timestep/2*tau)
	*
	* @param f distribution function
	* @return second moment of the distribution function f
	* using the array declared in structure
	*/
__device__ __forceinline__ struct structSecMomDistrFun _structCalculatePiTensor(const distribn_t* const f) //return type is struct structSecMomDistrFun
{
	struct structSecMomDistrFun ret_SecMomDistrFunc; //demo structure member declared

	// Fill the elements SecMomDistrFun.arr[i]; i=0 to 5
	// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
	// and saves these with this order in the struct array ret_SecMomDistrFunc.arr

	// Element (0,0)
	ret_SecMomDistrFunc.arr[0] = 0.0;
	for (unsigned int l = 0; l < _NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[0] += f[l] * _CX_19[l]* _CX_19[l];
	}

	// Element (1,0)
	ret_SecMomDistrFunc.arr[1] = 0.0;
	for (unsigned int l = 0; l < _NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[1] += f[l] * _CY_19[l]* _CX_19[l];
	}

	// Element (1,1)
	ret_SecMomDistrFunc.arr[2] = 0.0;
	for (unsigned int l = 0; l < _NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[2] += f[l] * _CY_19[l]* _CY_19[l];
	}

	// Element (2,0)
	ret_SecMomDistrFunc.arr[3] = 0.0;
	for (unsigned int l = 0; l < _NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[3] += f[l] * _CZ_19[l]* _CX_19[l];
	}

	// Element (2,1)
	ret_SecMomDistrFunc.arr[4] = 0.0;
	for (unsigned int l = 0; l < _NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[4] += f[l] * _CZ_19[l]* _CY_19[l];
	}

	// Element (2,2)
	ret_SecMomDistrFunc.arr[5] = 0.0;
	for (unsigned int l = 0; l < _NUMVECTORS; ++l)
	{
		ret_SecMomDistrFunc.arr[5] += f[l] * _CZ_19[l]* _CZ_19[l];
	}

	return ret_SecMomDistrFunc; //address of structure member returned
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
	* 	using a pointer to the array.
	* 		Note that the array needs to be declared as static, otherwise compiling issues might occur
	*/
	__device__ __forceinline__ double *_CalculatePiTensor(const distribn_t* const f)
	{
			static double ret_SecMomDistrFunc[6]; // Needs to be static

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
//==============================================================================

	__device__ __forceinline__ double _CalculateWallShearStressMagnitude(const distribn_t density,
			const distribn_t* const f_neq,
			const double normal_x, const double normal_y, const double normal_z,
			const double &iStressParameter)
	{
		distribn_t wall_shear_stress_magn;

		//printf("Wall normal components: (%5.5e, %5.5e, %5.5e)\n", normal_x, normal_y, normal_z);

		// sigma_ij is the force
		// per unit area in
		// direction i on the
		// plane with the normal
		// in direction j
		distribn_t stress_vector[] = { 0.0, 0.0, 0.0 }; // Force per unit area in
		// direction i on the
		// plane perpendicular to
		// the surface normal
		distribn_t square_stress_vector = 0.0;
		distribn_t normal_stress = 0.0; // Magnitude of force per
		// unit area normal to the
		// surface

		// Multiplying the second moment of the non equilibrium function by temp gives the non equilibrium part
		// of the moment flux tensor pi.
		distribn_t temp = iStressParameter * (-sqrt(2.0));

		// Computes the second moment of the argument passed ( non equilibrium part of f).
		// This will initially evaluate the second moments of the distr. functions
		// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
		// and saves these with this order in the array ret_SecMomDistrFunc
		// 	Need then to exploit symmetry to fill the elements (0,1) (0,2) (1,2)

		/*
		// Old approach
		double *SecMomDistrFunc;
		SecMomDistrFunc = _CalculatePiTensor(f_neq);
		*/

		double *SecMomDistrFunc;
		//--------------------------------------------------------------------------
		/*
		// Approach 1: Using a pointer to the array
		double *SecMomDistrFunc_returned;
		SecMomDistrFunc_returned = _CalculatePiTensor(f_neq);
		SecMomDistrFunc = SecMomDistrFunc_returned;
		*/

		// Approach 2: Using a struct and array declared in that struct
		struct structSecMomDistrFun SecMomDistrFunc_returned;
		SecMomDistrFunc_returned = _structCalculatePiTensor(f_neq);
		SecMomDistrFunc = SecMomDistrFunc_returned.arr;

		//--------------------------------------------------------------------------

		// Does not need the following - Use symmetry
		//SecMomDistrFunc[6] = SecMomDistrFunc[1];
		//SecMomDistrFunc[7] = SecMomDistrFunc[3];
		//SecMomDistrFunc[8] = SecMomDistrFunc[4];
		/*// Debugging
		for (int i = 0; i < 6; ++i) {
			printf("Second Mom. Distr. funct. %5.5e\n", SecMomDistrFunc[i]);
		}*/

		// Original loop:
		/*for (unsigned i = 0; i < 3; i++)
		{
			for (unsigned j = 0; j < 3; j++){
				stress_vector[i] += pi[i][j] * nor[j] * temp;
			}
			square_stress_vector += stress_vector[i] * stress_vector[i];

			//normal_stress += stress_vector[i] * nor[i];
		}
		*/

		// Unrolled loops:
		stress_vector[0] = ( SecMomDistrFunc[0] * normal_x +
												SecMomDistrFunc[1] * normal_y +
												SecMomDistrFunc[3] * normal_z) * temp;
		stress_vector[1] = ( SecMomDistrFunc[1] * normal_x +
												SecMomDistrFunc[2] * normal_y +
												SecMomDistrFunc[4] * normal_z) * temp;
		stress_vector[2] = ( SecMomDistrFunc[3] * normal_x +
												SecMomDistrFunc[4] * normal_y +
												SecMomDistrFunc[5] * normal_z) * temp;

		square_stress_vector = 	stress_vector[0] * stress_vector[0] +
														stress_vector[1] * stress_vector[1] +
														stress_vector[2] * stress_vector[2];

		normal_stress = 	stress_vector[0] * normal_x
										+ stress_vector[1] * normal_y
										+ stress_vector[2] * normal_z;

		// shear_stress^2 + normal_stress^2 = stress_vector^2
		//stress = sqrt(square_stress_vector - normal_stress * normal_stress);
		wall_shear_stress_magn = sqrt(square_stress_vector - normal_stress * normal_stress);;

		//printf("Wall Shear Stress = %5.5e, Sq.StressVect = %5.5e, NormStressSq = %5.5e\n", wall_shear_stress_magn, square_stress_vector, normal_stress * normal_stress);

		return wall_shear_stress_magn;
	}
	//==============================================================================

//==============================================================================
// Device Function that computes the LES relaxation time (localTau)
// tau0: Relaxation Time tau0(initParams.lbmParams->GetTau())
	__device__ __forceinline__ double _Compute_tau_smagorinsky(
		const distribn_t tau0,
		const distribn_t smag_cnst,
		const distribn_t* const f_neq)
	{
		/*double localTau;

		// calculate tau using smagorinsky local correction
		constexpr double dx = 1.0;
    constexpr double dt = 1.0;
    constexpr double C = dx / dt;
    constexpr double rho1 = 1.0;
    constexpr double C_smag = 0.1;

		double Q_12 = 0.0;

		// Calculate diagonal and upper diagonal of the non equilibrium stress tensor

		// i=0, j=0
		double qij_00 = 0.0;  // q_xx
		qij_00 += _CX_19[0] * _CX_19[0] * f_neq[0];
		qij_00 += _CX_19[1] * _CX_19[1] * f_neq[1];
		qij_00 += _CX_19[2] * _CX_19[2] * f_neq[2];
		qij_00 += _CX_19[3] * _CX_19[3] * f_neq[3];
		qij_00 += _CX_19[4] * _CX_19[4] * f_neq[4];
		qij_00 += _CX_19[5] * _CX_19[5] * f_neq[5];
		qij_00 += _CX_19[6] * _CX_19[6] * f_neq[6];
		qij_00 += _CX_19[7] * _CX_19[7] * f_neq[7];
		qij_00 += _CX_19[8] * _CX_19[8] * f_neq[8];
		qij_00 += _CX_19[9] * _CX_19[9] * f_neq[9];
		qij_00 += _CX_19[10] * _CX_19[10] * f_neq[10];
		qij_00 += _CX_19[11] * _CX_19[11] * f_neq[11];
		qij_00 += _CX_19[12] * _CX_19[12] * f_neq[12];
		qij_00 += _CX_19[13] * _CX_19[13] * f_neq[13];
		qij_00 += _CX_19[14] * _CX_19[14] * f_neq[14];
		qij_00 += _CX_19[15] * _CX_19[15] * f_neq[15];
		qij_00 += _CX_19[16] * _CX_19[16] * f_neq[16];
		qij_00 += _CX_19[17] * _CX_19[17] * f_neq[17];
		qij_00 += _CX_19[18] * _CX_19[18] * f_neq[18];

		// i=0, j=1
		double qij_01 = 0.0;  // q_xy
		qij_01 += _CX_19[0] * _CY_19[0] * f_neq[0];
		qij_01 += _CX_19[1] * _CY_19[1] * f_neq[1];
		qij_01 += _CX_19[2] * _CY_19[2] * f_neq[2];
		qij_01 += _CX_19[3] * _CY_19[3] * f_neq[3];
		qij_01 += _CX_19[4] * _CY_19[4] * f_neq[4];
		qij_01 += _CX_19[5] * _CY_19[5] * f_neq[5];
		qij_01 += _CX_19[6] * _CY_19[6] * f_neq[6];
		qij_01 += _CX_19[7] * _CY_19[7] * f_neq[7];
		qij_01 += _CX_19[8] * _CY_19[8] * f_neq[8];
		qij_01 += _CX_19[9] * _CY_19[9] * f_neq[9];
		qij_01 += _CX_19[10] * _CY_19[10] * f_neq[10];
		qij_01 += _CX_19[11] * _CY_19[11] * f_neq[11];
		qij_01 += _CX_19[12] * _CY_19[12] * f_neq[12];
		qij_01 += _CX_19[13] * _CY_19[13] * f_neq[13];
		qij_01 += _CX_19[14] * _CY_19[14] * f_neq[14];
		qij_01 += _CX_19[15] * _CY_19[15] * f_neq[15];
		qij_01 += _CX_19[16] * _CY_19[16] * f_neq[16];
		qij_01 += _CX_19[17] * _CY_19[17] * f_neq[17];
		qij_01 += _CX_19[18] * _CY_19[18] * f_neq[18];

		// i=0, j=2
		double qij_02 = 0.0;  // q_xz
		qij_02 += _CX_19[0] * _CZ_19[0] * f_neq[0];
		qij_02 += _CX_19[1] * _CZ_19[1] * f_neq[1];
		qij_02 += _CX_19[2] * _CZ_19[2] * f_neq[2];
		qij_02 += _CX_19[3] * _CZ_19[3] * f_neq[3];
		qij_02 += _CX_19[4] * _CZ_19[4] * f_neq[4];
		qij_02 += _CX_19[5] * _CZ_19[5] * f_neq[5];
		qij_02 += _CX_19[6] * _CZ_19[6] * f_neq[6];
		qij_02 += _CX_19[7] * _CZ_19[7] * f_neq[7];
		qij_02 += _CX_19[8] * _CZ_19[8] * f_neq[8];
		qij_02 += _CX_19[9] * _CZ_19[9] * f_neq[9];
		qij_02 += _CX_19[10] * _CZ_19[10] * f_neq[10];
		qij_02 += _CX_19[11] * _CZ_19[11] * f_neq[11];
		qij_02 += _CX_19[12] * _CZ_19[12] * f_neq[12];
		qij_02 += _CX_19[13] * _CZ_19[13] * f_neq[13];
		qij_02 += _CX_19[14] * _CZ_19[14] * f_neq[14];
		qij_02 += _CX_19[15] * _CZ_19[15] * f_neq[15];
		qij_02 += _CX_19[16] * _CZ_19[16] * f_neq[16];
		qij_02 += _CX_19[17] * _CZ_19[17] * f_neq[17];
		qij_02 += _CX_19[18] * _CZ_19[18] * f_neq[18];

		// i=1, j=1
		double qij_11 = 0.0;  // q_yy
		qij_11 += _CY_19[0] * _CY_19[0] * f_neq[0];
		qij_11 += _CY_19[1] * _CY_19[1] * f_neq[1];
		qij_11 += _CY_19[2] * _CY_19[2] * f_neq[2];
		qij_11 += _CY_19[3] * _CY_19[3] * f_neq[3];
		qij_11 += _CY_19[4] * _CY_19[4] * f_neq[4];
		qij_11 += _CY_19[5] * _CY_19[5] * f_neq[5];
		qij_11 += _CY_19[6] * _CY_19[6] * f_neq[6];
		qij_11 += _CY_19[7] * _CY_19[7] * f_neq[7];
		qij_11 += _CY_19[8] * _CY_19[8] * f_neq[8];
		qij_11 += _CY_19[9] * _CY_19[9] * f_neq[9];
		qij_11 += _CY_19[10] * _CY_19[10] * f_neq[10];
		qij_11 += _CY_19[11] * _CY_19[11] * f_neq[11];
		qij_11 += _CY_19[12] * _CY_19[12] * f_neq[12];
		qij_11 += _CY_19[13] * _CY_19[13] * f_neq[13];
		qij_11 += _CY_19[14] * _CY_19[14] * f_neq[14];
		qij_11 += _CY_19[15] * _CY_19[15] * f_neq[15];
		qij_11 += _CY_19[16] * _CY_19[16] * f_neq[16];
		qij_11 += _CY_19[17] * _CY_19[17] * f_neq[17];
		qij_11 += _CY_19[18] * _CY_19[18] * f_neq[18];

		// i=1, j=2
		double qij_12 = 0.0;  // q_yz
		qij_12 += _CY_19[0] * _CZ_19[0] * f_neq[0];
		qij_12 += _CY_19[1] * _CZ_19[1] * f_neq[1];
		qij_12 += _CY_19[2] * _CZ_19[2] * f_neq[2];
		qij_12 += _CY_19[3] * _CZ_19[3] * f_neq[3];
		qij_12 += _CY_19[4] * _CZ_19[4] * f_neq[4];
		qij_12 += _CY_19[5] * _CZ_19[5] * f_neq[5];
		qij_12 += _CY_19[6] * _CZ_19[6] * f_neq[6];
		qij_12 += _CY_19[7] * _CZ_19[7] * f_neq[7];
		qij_12 += _CY_19[8] * _CZ_19[8] * f_neq[8];
		qij_12 += _CY_19[9] * _CZ_19[9] * f_neq[9];
		qij_12 += _CY_19[10] * _CZ_19[10] * f_neq[10];
		qij_12 += _CY_19[11] * _CZ_19[11] * f_neq[11];
		qij_12 += _CY_19[12] * _CZ_19[12] * f_neq[12];
		qij_12 += _CY_19[13] * _CZ_19[13] * f_neq[13];
		qij_12 += _CY_19[14] * _CZ_19[14] * f_neq[14];
		qij_12 += _CY_19[15] * _CZ_19[15] * f_neq[15];
		qij_12 += _CY_19[16] * _CZ_19[16] * f_neq[16];
		qij_12 += _CY_19[17] * _CZ_19[17] * f_neq[17];
		qij_12 += _CY_19[18] * _CZ_19[18] * f_neq[18];

		// i=2, j=2
		double qij_22 = 0.0;  // q_zz
		qij_22 += _CZ_19[0] * _CZ_19[0] * f_neq[0];
		qij_22 += _CZ_19[1] * _CZ_19[1] * f_neq[1];
		qij_22 += _CZ_19[2] * _CZ_19[2] * f_neq[2];
		qij_22 += _CZ_19[3] * _CZ_19[3] * f_neq[3];
		qij_22 += _CZ_19[4] * _CZ_19[4] * f_neq[4];
		qij_22 += _CZ_19[5] * _CZ_19[5] * f_neq[5];
		qij_22 += _CZ_19[6] * _CZ_19[6] * f_neq[6];
		qij_22 += _CZ_19[7] * _CZ_19[7] * f_neq[7];
		qij_22 += _CZ_19[8] * _CZ_19[8] * f_neq[8];
		qij_22 += _CZ_19[9] * _CZ_19[9] * f_neq[9];
		qij_22 += _CZ_19[10] * _CZ_19[10] * f_neq[10];
		qij_22 += _CZ_19[11] * _CZ_19[11] * f_neq[11];
		qij_22 += _CZ_19[12] * _CZ_19[12] * f_neq[12];
		qij_22 += _CZ_19[13] * _CZ_19[13] * f_neq[13];
		qij_22 += _CZ_19[14] * _CZ_19[14] * f_neq[14];
		qij_22 += _CZ_19[15] * _CZ_19[15] * f_neq[15];
		qij_22 += _CZ_19[16] * _CZ_19[16] * f_neq[16];
		qij_22 += _CZ_19[17] * _CZ_19[17] * f_neq[17];
		qij_22 += _CZ_19[18] * _CZ_19[18] * f_neq[18];

		// Sum the terms to get Q_12 (loops over i and j -> ie 9 terms qij * qij)
		Q_12 += qij_00 * qij_00 + qij_01 * qij_01 + qij_02 * qij_02;
		Q_12 += qij_01 * qij_01 + qij_11 * qij_11 + qij_12 * qij_12;
		Q_12 += qij_02 * qij_02 + qij_12 * qij_12 + qij_22 * qij_22;

		Q_12 = sqrt(Q_12);

		// eq 36 Koda 2015, csmag is smagorinsky constant here is c_smag^2 in the paper is c_smag
		localTau = 1. / 2. *
									(tau0 + sqrt((tau0 * rho1 * C)*(tau0 * rho1 * C) +
									18.0 * 1.4142135623730950488016887242097 * rho1 * C_smag * C_smag * Q_12) / (rho1 * C));

		return localTau;
		*/
		// Constants (should be precomputed if possible)
    constexpr double dx = 1.0;
    constexpr double dt = 1.0;
    constexpr double C = dx / dt;
    constexpr double rho1 = 1.0;
    double C_smag = smag_cnst; // constexpr double C_smag = 0.1;

    double Q_12 = 0.0;

    // Compute qij values
    const double qij_00 =
        _CX_19[0] * _CX_19[0] * f_neq[0] + _CX_19[1] * _CX_19[1] * f_neq[1] +
        _CX_19[2] * _CX_19[2] * f_neq[2] + _CX_19[3] * _CX_19[3] * f_neq[3] +
        _CX_19[4] * _CX_19[4] * f_neq[4] + _CX_19[5] * _CX_19[5] * f_neq[5] +
        _CX_19[6] * _CX_19[6] * f_neq[6] + _CX_19[7] * _CX_19[7] * f_neq[7] +
        _CX_19[8] * _CX_19[8] * f_neq[8] + _CX_19[9] * _CX_19[9] * f_neq[9] +
        _CX_19[10] * _CX_19[10] * f_neq[10] + _CX_19[11] * _CX_19[11] * f_neq[11] +
        _CX_19[12] * _CX_19[12] * f_neq[12] + _CX_19[13] * _CX_19[13] * f_neq[13] +
        _CX_19[14] * _CX_19[14] * f_neq[14] + _CX_19[15] * _CX_19[15] * f_neq[15] +
        _CX_19[16] * _CX_19[16] * f_neq[16] + _CX_19[17] * _CX_19[17] * f_neq[17] +
        _CX_19[18] * _CX_19[18] * f_neq[18];

    const double qij_01 =
        _CX_19[0] * _CY_19[0] * f_neq[0] + _CX_19[1] * _CY_19[1] * f_neq[1] +
        _CX_19[2] * _CY_19[2] * f_neq[2] + _CX_19[3] * _CY_19[3] * f_neq[3] +
        _CX_19[4] * _CY_19[4] * f_neq[4] + _CX_19[5] * _CY_19[5] * f_neq[5] +
        _CX_19[6] * _CY_19[6] * f_neq[6] + _CX_19[7] * _CY_19[7] * f_neq[7] +
        _CX_19[8] * _CY_19[8] * f_neq[8] + _CX_19[9] * _CY_19[9] * f_neq[9] +
        _CX_19[10] * _CY_19[10] * f_neq[10] + _CX_19[11] * _CY_19[11] * f_neq[11] +
        _CX_19[12] * _CY_19[12] * f_neq[12] + _CX_19[13] * _CY_19[13] * f_neq[13] +
        _CX_19[14] * _CY_19[14] * f_neq[14] + _CX_19[15] * _CY_19[15] * f_neq[15] +
        _CX_19[16] * _CY_19[16] * f_neq[16] + _CX_19[17] * _CY_19[17] * f_neq[17] +
        _CX_19[18] * _CY_19[18] * f_neq[18];

    const double qij_02 =
        _CX_19[0] * _CZ_19[0] * f_neq[0] + _CX_19[1] * _CZ_19[1] * f_neq[1] +
        _CX_19[2] * _CZ_19[2] * f_neq[2] + _CX_19[3] * _CZ_19[3] * f_neq[3] +
        _CX_19[4] * _CZ_19[4] * f_neq[4] + _CX_19[5] * _CZ_19[5] * f_neq[5] +
        _CX_19[6] * _CZ_19[6] * f_neq[6] + _CX_19[7] * _CZ_19[7] * f_neq[7] +
        _CX_19[8] * _CZ_19[8] * f_neq[8] + _CX_19[9] * _CZ_19[9] * f_neq[9] +
        _CX_19[10] * _CZ_19[10] * f_neq[10] + _CX_19[11] * _CZ_19[11] * f_neq[11] +
        _CX_19[12] * _CZ_19[12] * f_neq[12] + _CX_19[13] * _CZ_19[13] * f_neq[13] +
        _CX_19[14] * _CZ_19[14] * f_neq[14] + _CX_19[15] * _CZ_19[15] * f_neq[15] +
        _CX_19[16] * _CZ_19[16] * f_neq[16] + _CX_19[17] * _CZ_19[17] * f_neq[17] +
        _CX_19[18] * _CZ_19[18] * f_neq[18];

    const double qij_11 =
        _CY_19[0] * _CY_19[0] * f_neq[0] + _CY_19[1] * _CY_19[1] * f_neq[1] +
        _CY_19[2] * _CY_19[2] * f_neq[2] + _CY_19[3] * _CY_19[3] * f_neq[3] +
        _CY_19[4] * _CY_19[4] * f_neq[4] + _CY_19[5] * _CY_19[5] * f_neq[5] +
        _CY_19[6] * _CY_19[6] * f_neq[6] + _CY_19[7] * _CY_19[7] * f_neq[7] +
        _CY_19[8] * _CY_19[8] * f_neq[8] + _CY_19[9] * _CY_19[9] * f_neq[9] +
        _CY_19[10] * _CY_19[10] * f_neq[10] + _CY_19[11] * _CY_19[11] * f_neq[11] +
        _CY_19[12] * _CY_19[12] * f_neq[12] + _CY_19[13] * _CY_19[13] * f_neq[13] +
        _CY_19[14] * _CY_19[14] * f_neq[14] + _CY_19[15] * _CY_19[15] * f_neq[15] +
        _CY_19[16] * _CY_19[16] * f_neq[16] + _CY_19[17] * _CY_19[17] * f_neq[17] +
        _CY_19[18] * _CY_19[18] * f_neq[18];

    const double qij_12 =
        _CY_19[0] * _CZ_19[0] * f_neq[0] + _CY_19[1] * _CZ_19[1] * f_neq[1] +
        _CY_19[2] * _CZ_19[2] * f_neq[2] + _CY_19[3] * _CZ_19[3] * f_neq[3] +
        _CY_19[4] * _CZ_19[4] * f_neq[4] + _CY_19[5] * _CZ_19[5] * f_neq[5] +
        _CY_19[6] * _CZ_19[6] * f_neq[6] + _CY_19[7] * _CZ_19[7] * f_neq[7] +
        _CY_19[8] * _CZ_19[8] * f_neq[8] + _CY_19[9] * _CZ_19[9] * f_neq[9] +
        _CY_19[10] * _CZ_19[10] * f_neq[10] + _CY_19[11] * _CZ_19[11] * f_neq[11] +
        _CY_19[12] * _CZ_19[12] * f_neq[12] + _CY_19[13] * _CZ_19[13] * f_neq[13] +
        _CY_19[14] * _CZ_19[14] * f_neq[14] + _CY_19[15] * _CZ_19[15] * f_neq[15] +
        _CY_19[16] * _CZ_19[16] * f_neq[16] + _CY_19[17] * _CZ_19[17] * f_neq[17] +
        _CY_19[18] * _CZ_19[18] * f_neq[18];

    const double qij_22 =
        _CZ_19[0] * _CZ_19[0] * f_neq[0] + _CZ_19[1] * _CZ_19[1] * f_neq[1] +
        _CZ_19[2] * _CZ_19[2] * f_neq[2] + _CZ_19[3] * _CZ_19[3] * f_neq[3] +
        _CZ_19[4] * _CZ_19[4] * f_neq[4] + _CZ_19[5] * _CZ_19[5] * f_neq[5] +
        _CZ_19[6] * _CZ_19[6] * f_neq[6] + _CZ_19[7] * _CZ_19[7] * f_neq[7] +
        _CZ_19[8] * _CZ_19[8] * f_neq[8] + _CZ_19[9] * _CZ_19[9] * f_neq[9] +
        _CZ_19[10] * _CZ_19[10] * f_neq[10] + _CZ_19[11] * _CZ_19[11] * f_neq[11] +
        _CZ_19[12] * _CZ_19[12] * f_neq[12] + _CZ_19[13] * _CZ_19[13] * f_neq[13] +
        _CZ_19[14] * _CZ_19[14] * f_neq[14] + _CZ_19[15] * _CZ_19[15] * f_neq[15] +
        _CZ_19[16] * _CZ_19[16] * f_neq[16] + _CZ_19[17] * _CZ_19[17] * f_neq[17] +
        _CZ_19[18] * _CZ_19[18] * f_neq[18];

    // Sum the terms to get Q_12 (loops over i and j -> ie 9 terms qij * qij)
    Q_12 += qij_00 * qij_00 + qij_01 * qij_01 + qij_02 * qij_02;
    Q_12 += qij_01 * qij_01 + qij_11 * qij_11 + qij_12 * qij_12;
    Q_12 += qij_02 * qij_02 + qij_12 * qij_12 + qij_22 * qij_22;

    Q_12 = sqrt(Q_12);

    // Calculate the local Tau using the given formula
    return 0.5 * (tau0 + sqrt((tau0 * rho1 * C) * (tau0 * rho1 * C) +
                              18.0 * 1.4142135623730950488016887242097 * rho1 * C_smag * C_smag * Q_12) / (rho1 * C));

}

//==============================================================================
/**
* Device Function that computes the relaxation time to be used in the
* evolution equation for the distr. functions (collision)
* The sponge layer is maintained for a certain time and then dissolved.
*/
	__device__ __forceinline__ double _CalculateTau(
		const distribn_t tau0,
		const distribn_t smag_cnst,
		const distribn_t vTau_local,
		const unsigned long timeStep,
		const unsigned long lifetime,
		const distribn_t* const f_neq)
	{

		/*double ret_tau; // The relaxation time to be returned

		double tau_les = _Compute_tau_smagorinsky(tau0, f_neq);
		//printf("tau_les: %f , inv tau_les: %f  \n", tau_les, 1./tau_les);
		if (timeStep <= lifetime / 2)
		{
			if(vTau_local == tau0){
				ret_tau = tau_les;
			}else{
				ret_tau = vTau_local;
			}
		}
		else if (timeStep < lifetime)
		{
			// Linear decay from vTau to tau0
			// hydroVars.tau = (tau0 - vTau[index]) * 2.0 / lifetime * timeStep + (2.0 * vTau[index] - tau0);
			//hydroVars.tau = (tau_les - vTau[index]) * 2.0 / lifetime * timeStep + (2.0 * vTau[index] - tau_les);
			ret_tau = (tau_les - vTau_local) * 2.0 / lifetime * timeStep + (2.0 * vTau_local - tau_les);
		}
		else
		{
			// hydroVars.tau = tau0;
			ret_tau = tau_les;
		}
		//printf("Ret_tau: %f, tau_les: %f \n", ret_tau, tau_les );

		return ret_tau;
		*/

		// Calculate tau_les using the Smagorinsky model
    double tau_les = _Compute_tau_smagorinsky(tau0, smag_cnst, f_neq);

    // Determine the relaxation time based on the timeStep and lifetime
    if (timeStep <= lifetime / 2)
    {
        return (vTau_local == tau0) ? tau_les : vTau_local;
    }
    else if (timeStep < lifetime)
    {
        return (tau_les - vTau_local) * 2.0 / lifetime * timeStep + (2.0 * vTau_local - tau_les);
    }
    else
    {
        return tau_les;
    }
	}



}
#endif
