// This file is part of HemeLB and is Copyright (C)
// the HemeLB team and/or their institutions, as detailed in the
// file AUTHORS. This software is provided under the terms of the
// license in the file LICENSE.

#ifndef HEMELB_LB_LB_H
#define HEMELB_LB_LB_H

#include "net/net.h"
#include "net/IteratedAction.h"
#include "net/IOCommunicator.h"
#include "lb/SimulationState.h"
#include "lb/iolets/BoundaryValues.h"
#include "lb/MacroscopicPropertyCache.h"
#include "util/UnitConverter.h"
#include "configuration/SimConfig.h"
#include "reporting/Timers.h"
#include "lb/BuildSystemInterface.h"
#include <typeinfo>

// Maybe this is not needed
#include "net/MpiCommunicator.h"

// IZ
#ifdef HEMELB_USE_GPU
#include "cuda_kernels_def_decl/cuda_params.h"
#endif
// IZ


namespace hemelb
{
	/**
	 * Namespace 'lb' contains classes for the scientific core of the Lattice Boltzman simulation
	 */
	namespace lb
	{
		/**
		 * Class providing core Lattice Boltzmann functionality.
		 * Implements the IteratedAction interface.
		 */
		template<class LatticeType>
			class LBM : public net::IteratedAction
		{
				// IZ - Nov 2023 - Added for the Checkpointing functionality
				friend class extraction::PropertyActor; //! Give access to the boolean variable checkpointing_Get_Distr_To_Host
								//
			private:
				// Use the kernel specified through the build system. This will select one of the above classes.
				typedef typename HEMELB_KERNEL<LatticeType>::Type LB_KERNEL;

				typedef streamers::SimpleCollideAndStream<collisions::Normal<LB_KERNEL> > tMidFluidCollision;
				// Use the wall boundary condition specified through the build system.
				typedef typename HEMELB_WALL_BOUNDARY<collisions::Normal<LB_KERNEL> >::Type tWallCollision;
				// Use the inlet BC specified by the build system
				typedef typename HEMELB_INLET_BOUNDARY<collisions::Normal<LB_KERNEL> >::Type tInletCollision;
				// Use the outlet BC specified by the build system
				typedef typename HEMELB_OUTLET_BOUNDARY<collisions::Normal<LB_KERNEL> >::Type tOutletCollision;
				// And again but for sites that are both in-/outlet and wall
				typedef typename HEMELB_WALL_INLET_BOUNDARY<collisions::Normal<LB_KERNEL> >::Type tInletWallCollision;
				typedef typename HEMELB_WALL_OUTLET_BOUNDARY<collisions::Normal<LB_KERNEL> >::Type tOutletWallCollision;

			public:
				/**
				 * Constructor, stage 1.
				 * Object so initialized is not ready for simulation.
				 * Must have Initialise(...) called also. Constructor separated due to need to access
				 * the partially initialized LBM in order to initialize the arguments to the second construction phase.
				 */
				LBM(hemelb::configuration::SimConfig *iSimulationConfig,
						net::Net* net,
						geometry::LatticeData* latDat,
						SimulationState* simState,
						reporting::Timers &atimings,
						geometry::neighbouring::NeighbouringDataManager *neighbouringDataManager);
				~LBM();

				void RequestComms(); ///< part of IteratedAction interface.
				void PreSend(); ///< part of IteratedAction interface.
				void PreReceive(); ///< part of IteratedAction interface.
				void PostReceive(); ///< part of IteratedAction interface.
				void EndIteration(); ///< part of IteratedAction interface.

				site_t TotalFluidSiteCount() const;
				void SetTotalFluidSiteCount(site_t);
				int InletCount() const
				{
					return inletCount;
				}
				int OutletCount() const
				{
					return outletCount;
				}

				/**
				 * Second constructor.
				 */
				void Initialise(iolets::BoundaryValues* iInletValues,
						iolets::BoundaryValues* iOutletValues,
						const util::UnitConverter* iUnits);

				void SetInitialConditions(const net::IOCommunicator& ioComms);

				hemelb::lb::LbmParameters *GetLbmParams();
				lb::MacroscopicPropertyCache& GetPropertyCache();

				//IZ
//========================================================================
#ifdef HEMELB_USE_GPU
				//GPU Data Addresses - Remove later the ones not used - See memory allocations for the distrib. functions (method a and b)
				void *GPUDataAddr_dbl_fOld, *GPUDataAddr_dbl_fNew;
				void *GPUDataAddr_dbl_MacroVars;
				void *GPUDataAddr_int64_Neigh;
				void *GPUDataAddr_int64_streamInd;
				void *GPUDataAddr_uint32_Wall;
				void *GPUDataAddr_uint32_Iolet;

				void *d_ghostDensity, *d_inletNormal;	// ghostDensity and inlet Normals
				void *d_ghostDensity_out, *d_outletNormal;	// ghostDensity and inlet Normals

				void *GPUDataAddr_dbl_fOld_b, *GPUDataAddr_dbl_fNew_b;
				void *GPUDataAddr_int64_Neigh_b;
				void *GPUDataAddr_int64_Neigh_c;
				void *GPUDataAddr_int64_Neigh_d;

				void *GPUDataAddr_vTau = nullptr;
				//--------------------------------------------------
				// Vel. BCs case - Transfer everything on the GPU and compute the wall momentum correction on the GPU
				void **GPUDataAddr_pp_Inlet_weightsTable_coord = nullptr; // Pointer to pointers
				void *GPUDataAddr_p_Inlet_weightsTable_coord = nullptr; // void *GPUDataAddr_p_Inlet_weightsTable_coord;

				void **CPU_DataAddr_pp_Inlet_weightsTable_coord = nullptr; // Holds the address of the pointers to the GPU global memory (pinned memory - allocated with cudaMallocHost, so that it can be accessed by the device directly)
				void *GPUDataAddr_p_Inlet_weightsTable_coord_x, *GPUDataAddr_p_Inlet_weightsTable_coord_y, *GPUDataAddr_p_Inlet_weightsTable_coord_z;


				void **GPUDataAddr_pp_Inlet_weightsTable_wei = nullptr;	// Pointer to pointers
				distribn_t *GPUDataAddr_p_Inlet_weightsTable_wei = nullptr;
				//--------------------------------------------------

				// Iolets Info: Used for the case of Pressure BCs (NASHZEROTHORDERPRESSUREIOLET) - Vel BCs as well
				void *GPUDataAddr_Inlet_Edge, *GPUDataAddr_Outlet_Edge, *GPUDataAddr_InletWall_Edge, *GPUDataAddr_OutletWall_Edge;
				void *GPUDataAddr_Inlet_Inner, *GPUDataAddr_Outlet_Inner, *GPUDataAddr_InletWall_Inner, *GPUDataAddr_OutletWall_Inner;

				// wall Momentum associated with Velocity BCs (LADDIOLET) - GPU global memory related
				void *GPUDataAddr_wallMom_Inlet_Edge;
				void *GPUDataAddr_wallMom_InletWall_Edge;
				void *GPUDataAddr_wallMom_Inlet_Inner;
				void *GPUDataAddr_wallMom_InletWall_Inner;
				void *GPUDataAddr_wallMom_Outlet_Edge;
				void *GPUDataAddr_wallMom_OutletWall_Edge;
				void *GPUDataAddr_wallMom_Outlet_Inner;
				void *GPUDataAddr_wallMom_OutletWall_Inner;

				//---------------------------------------------------
				// Case of Velocity BCs - Subtype: Case b. File
				distribn_t *Data_dbl_Inlet_velocityTable, *Data_dbl_Outlet_velocityTable;
				int *arr_elementsInEachInlet;

				//------------------------------------------------------------------
				// BUG in the approach below - Remnove what is below as soos as the following
				//		approach work:
				//---------------------
				// Approach with the bug: Remove what is below later
				// Array with the index in the weights table to obtain the weight (arranged based on the fluid index)
				/*
				int *index_weightTable_Inlet_Edge;
				int *index_weightTable_InletWall_Edge;
				int *index_weightTable_Inlet_Inner;
				int *index_weightTable_InletWall_Inner;
				*/
				std::vector<int64_t> index_weightTable_Inlet_Edge;
				std::vector<int64_t> index_weightTable_InletWall_Edge;
				std::vector<int64_t> index_weightTable_InletWall_Inner;
				std::vector<int64_t> index_weightTable_Inlet_Inner;

				// Corresponding GPU Data Addresses for the indices
				void *GPUDataAddr_index_weightTable_Inlet_Edge = nullptr;
				void *GPUDataAddr_index_weightTable_InletWall_Edge = nullptr;
				void *GPUDataAddr_index_weightTable_Inlet_Inner = nullptr;
				void *GPUDataAddr_index_weightTable_InletWall_Inner = nullptr;
				//---------------------

				// New Approah: Save ONLY the Vel. Weight directly to the GPU global memory BASED ON FLUID INDEX
				// 							Not the index and not the whole vel Weights Table
				std::vector<distribn_t> weightTable_Inlet_Edge;
				std::vector<distribn_t> weightTable_InletWall_Edge;
				std::vector<distribn_t> weightTable_InletWall_Inner;
				std::vector<distribn_t> weightTable_Inlet_Inner;

				// Corresponding GPU Data Addresses for the VEL WEIGHTS
				void *GPUDataAddr_weightTable_Inlet_Edge;
				void *GPUDataAddr_weightTable_InletWall_Edge;
				void *GPUDataAddr_weightTable_Inlet_Inner;
				void *GPUDataAddr_weightTable_InletWall_Inner;

				//------------------------------------------------------------------
				//---------------------------------------------------
				//std::vector<int> arr_elementsInEachInlet;
				//thrust::host_vector<int> arr_elementsInEachInlet(1);

				// And the corresponding host vectors related to the above
				std::vector<util::Vector3D<double> > wallMom_Inlet_Edge;
				std::vector<util::Vector3D<double> > wallMom_InletWall_Edge;
				std::vector<util::Vector3D<double> > wallMom_Inlet_Inner;
				std::vector<util::Vector3D<double> > wallMom_InletWall_Inner;
				std::vector<util::Vector3D<double> > wallMom_Outlet_Edge;
				std::vector<util::Vector3D<double> > wallMom_OutletWall_Edge;
				std::vector<util::Vector3D<double> > wallMom_Outlet_Inner;
				std::vector<util::Vector3D<double> > wallMom_OutletWall_Inner;

				//----------------------------------------------------------------------
				// Work in progress 9 March 2022 - Done!!!
				// Instead of using wall momentum - just pass the correction term (one value instead of 3)
				// correction (or wall Momentum) associated with Velocity BCs (LADDIOLET) - GPU global memory related
				// Still too slow
				void *GPUDataAddr_wallMom_correction_Inlet_Edge = nullptr;
				void *GPUDataAddr_wallMom_correction_InletWall_Edge = nullptr;
				void *GPUDataAddr_wallMom_correction_Inlet_Inner = nullptr;
				void *GPUDataAddr_wallMom_correction_InletWall_Inner = nullptr;
				void *GPUDataAddr_wallMom_correction_Outlet_Edge = nullptr;
				void *GPUDataAddr_wallMom_correction_OutletWall_Edge = nullptr;
				void *GPUDataAddr_wallMom_correction_Outlet_Inner = nullptr;
				void *GPUDataAddr_wallMom_correction_OutletWall_Inner = nullptr;

				// And the corresponding host vectors related to the above
				// Replace the above with a single correction term instead of 3 components
				std::vector<distribn_t> wallMom_correction_Inlet_Edge;
				std::vector<distribn_t> wallMom_correction_InletWall_Edge;
				std::vector<distribn_t> wallMom_correction_Inlet_Inner;
				std::vector<distribn_t> wallMom_correction_InletWall_Inner;
				std::vector<distribn_t> wallMom_correction_Outlet_Edge;
				std::vector<distribn_t> wallMom_correction_OutletWall_Edge;
				std::vector<distribn_t> wallMom_correction_Outlet_Inner;
				std::vector<distribn_t> wallMom_correction_OutletWall_Inner;
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Work in progress July 2022 - Done!!!
				// Instead of passing the correction term to the Cache and then reading back
				// Directly Get the correction wall Momentum associated with Velocity BCs (LADDIOLET) - GPU global memory related
				void *GPUDataAddr_wallMom_correction_Inlet_Edge_Direct;
				void *GPUDataAddr_wallMom_correction_InletWall_Edge_Direct;
				void *GPUDataAddr_wallMom_correction_Inlet_Inner_Direct;
				void *GPUDataAddr_wallMom_correction_InletWall_Inner_Direct;
				void *GPUDataAddr_wallMom_correction_Outlet_Edge_Direct;
				void *GPUDataAddr_wallMom_correction_OutletWall_Edge_Direct;
				void *GPUDataAddr_wallMom_correction_Outlet_Inner_Direct;
				void *GPUDataAddr_wallMom_correction_OutletWall_Inner_Direct;

				// And the corresponding host vectors related to the above
				// Replace the above with a single correction term instead of 3 components
				std::vector<distribn_t> wallMom_correction_Inlet_Edge_Direct;
				std::vector<distribn_t> wallMom_correction_InletWall_Edge_Direct;
				std::vector<distribn_t> wallMom_correction_Inlet_Inner_Direct;
				std::vector<distribn_t> wallMom_correction_InletWall_Inner_Direct;
				std::vector<distribn_t> wallMom_correction_Outlet_Edge_Direct;
				std::vector<distribn_t> wallMom_correction_OutletWall_Edge_Direct;
				std::vector<distribn_t> wallMom_correction_Outlet_Inner_Direct;
				std::vector<distribn_t> wallMom_correction_OutletWall_Inner_Direct;

				std::vector<distribn_t> wallMom_correction_ColType_Domain_Direct;
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Work in progress April 2023 - TODO!!!
				// 	Instead of passing the wall momentum correction terms to the GPU at each time-step
				// 	Directly pass this prefactor at initialisation to the GPU and
				//		evaluate the wall momentum correction terms on the GPUs
				// 	Related to wall Momentum - Velocity BCs (LADDIOLET) - GPU global memory related
				void *GPUDataAddr_wallMom_prefactor_correction_Inlet_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_InletWall_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_Inlet_Inner;
				void *GPUDataAddr_wallMom_prefactor_correction_InletWall_Inner;
				void *GPUDataAddr_wallMom_prefactor_correction_Outlet_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_OutletWall_Edge;
				void *GPUDataAddr_wallMom_prefactor_correction_Outlet_Inner;
				void *GPUDataAddr_wallMom_prefactor_correction_OutletWall_Inner;

				// And the corresponding host vectors related to the above
				// Replace the above with a single correction term instead of 3 components
				std::vector<distribn_t> wallMom_prefactor_correction_Inlet_Edge;
				std::vector<distribn_t> wallMom_prefactor_correction_InletWall_Edge;
				std::vector<distribn_t> wallMom_prefactor_correction_Inlet_Inner;
				std::vector<distribn_t> wallMom_prefactor_correction_InletWall_Inner;
				std::vector<distribn_t> wallMom_prefactor_correction_Outlet_Edge;
				std::vector<distribn_t> wallMom_prefactor_correction_OutletWall_Edge;
				std::vector<distribn_t> wallMom_prefactor_correction_Outlet_Inner;
				std::vector<distribn_t> wallMom_prefactor_correction_OutletWall_Inner;

				std::vector<distribn_t> wallMom_prefactor_correction_ColType_Domain;
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Work in progress Oct 2022 - Done!!!
				// Send the iolets coords and the fluid index (fluid index, x_coord, y_coord, z_coord)
				// to the GPU global memory (type site_t which is int64_t)
				void *GPUDataAddr_Coords_Inlet_Edge;
				void *GPUDataAddr_Coords_InletWall_Edge;
				void *GPUDataAddr_Coords_Inlet_Inner;
				void *GPUDataAddr_Coords_InletWall_Inner;
				void *GPUDataAddr_Coords_Outlet_Edge;
				void *GPUDataAddr_Coords_OutletWall_Edge;
				void *GPUDataAddr_Coords_Outlet_Inner;
				void *GPUDataAddr_Coords_OutletWall_Inner;

				void *GPUDataAddr_inlets_position;
				void *GPUDataAddr_outlets_position;

				void *GPUDataAddr_inlets_radius;
				void *GPUDataAddr_outlets_radius;
				//----------------------------------------------------------------------

				//----------------------------------------------------------------------
				// Wall Shear Stress Magnitude - Type refers to collision types
				void *GPUDataAddr_WallShearStressMagn_Edge_Type2;
				void *GPUDataAddr_WallShearStressMagn_Edge_Type5;
				void *GPUDataAddr_WallShearStressMagn_Edge_Type6;
				void *GPUDataAddr_WallShearStressMagn_Inner_Type2;
				void *GPUDataAddr_WallShearStressMagn_Inner_Type5;
				void *GPUDataAddr_WallShearStressMagn_Inner_Type6;

				// Wall normal vector
				void *GPUDataAddr_WallNormal_Edge_Type2;
				void *GPUDataAddr_WallNormal_Edge_Type5;
				void *GPUDataAddr_WallNormal_Edge_Type6;
				void *GPUDataAddr_WallNormal_Inner_Type2;
				void *GPUDataAddr_WallNormal_Inner_Type5;
				void *GPUDataAddr_WallNormal_Inner_Type6;
				//----------------------------------------------------------------------


				// Need to distinguish: (a) n_LocalInlets... Vs  (b) n_unique_LocalInlets... :
				// 		(a) is the one needed for the array with the Range of fluid sites for each iolet
				//		(b) is the unique number of iolets on the local Rank
				std::vector<site_t> Iolets_Inlet_Edge; 			// vector with Inlet IDs and range associated with PreSend collision-streaming Type 3 (mInletCollision)
				int n_LocalInlets_mInlet_Edge; 							// number of local Inlets involved during the PreSend mInletCollision collision - needed for the range of fluid sites involved
				int n_unique_LocalInlets_mInlet_Edge;				// number of unique local Inlets

				std::vector<site_t> Iolets_InletWall_Edge;	// vector with Inlet IDs and range associated with PreSend collision-streaming Type 5 (mInletWallCollision)
				int n_LocalInlets_mInletWall_Edge; 					// number of local Inlets involved during the PreSend mInletWallCollision collision
				int n_unique_LocalInlets_mInletWall_Edge;

				std::vector<site_t> Iolets_Inlet_Inner;			// vector with Inlet IDs and range associated with PreReceive collision-streaming Types 3 (mInletCollision)
				int n_LocalInlets_mInlet; 									// number of local Inlets involved during the PreReceive mInletCollision collision
				int n_unique_LocalInlets_mInlet;

				std::vector<site_t> Iolets_InletWall_Inner;	// vector with Inlet IDs and range associated with PreReceive collision-streaming Types 5 (mInletWallCollision)
				int n_LocalInlets_mInletWall; 							// number of local Inlets involved during the PreReceive mInletWallCollision collision
				int n_unique_LocalInlets_mInletWall;

				std::vector<site_t> Iolets_Outlet_Edge;			// vector with Outlet IDs and range associated with PreSend collision-streaming Types 4 (mOutletCollision)
				int n_LocalOutlets_mOutlet_Edge; 						// number of local Outlets involved during the PreSend mOutletCollision collision
				int n_unique_LocalOutlets_mOutlet_Edge;

				std::vector<site_t> Iolets_OutletWall_Edge;	// vector with Outlet IDs and range associated with PreSend collision-streaming Types 6 (mOutletWallCollision)
				int n_LocalOutlets_mOutletWall_Edge; 				// number of local Outlets involved during the PreSend mOutletWallCollision collision
				int n_unique_LocalOutlets_mOutletWall_Edge;

				std::vector<site_t> Iolets_Outlet_Inner;			// vector with Outlet IDs and range associated with PreReceive collision-streaming Types 4 (mOutletCollision)
				int n_LocalOutlets_mOutlet; 									// number of local Outlets involved during the PreReceive mOutletCollision collision
				int n_unique_LocalOutlets_mOutlet;

				std::vector<site_t> Iolets_OutletWall_Inner;	// vector with Outlet IDs and range associated with PreReceive collision-streaming Types 6 (mOutletWallCollision)
				int n_LocalOutlets_mOutletWall; 							// number of local Outlets involved during the PreReceive mOutletWallCollision collision
				int n_unique_LocalOutlets_mOutletWall;

				// struct Iolets defined in file cuda_kernels_def_decl/cuda_params.h
				struct hemelb::Iolets Inlet_Edge, Inlet_Inner, InletWall_Edge, InletWall_Inner;
				struct hemelb::Iolets Outlet_Edge, Outlet_Inner, OutletWall_Edge, OutletWall_Inner;

				// Pointer to Stability flag (type int*)
				void* d_Stability_GPU;
				//int* d_Stability_GPU;
				int h_Stability_GPU;

				// Pointer to pinned memory
				distribn_t *Data_H2D_memcpy_totalSharedFs, *Data_D2H_memcpy_totalSharedFs;

				// Declare Static Pointers for pinned memory used in Read_Macrovariables_GPU_to_CPU
				static distribn_t* dens_GPU;
				static distribn_t* vx_GPU;
				static distribn_t* vy_GPU;
				static distribn_t* vz_GPU;

				static distribn_t *WallShearStressMagn_Edge_Type2_GPU, *WallShearStressMagn_Edge_Type5_GPU, *WallShearStressMagn_Edge_Type6_GPU;
				static distribn_t *WallShearStressMagn_Inner_Type2_GPU, *WallShearStressMagn_Inner_Type5_GPU, *WallShearStressMagn_Inner_Type6_GPU;

				// Declare static pointers for pinned memory used for the ghost density (case of Pressure BCs)
				static distribn_t* h_ghostDensity_inlet;
				static distribn_t* h_ghostDensity_outlet;

				// Declare the Streams
				Stream_t Collide_Stream_PreSend_1, Collide_Stream_PreSend_2, Collide_Stream_PreSend_3, Collide_Stream_PreSend_4, Collide_Stream_PreSend_5, Collide_Stream_PreSend_6;
				Stream_t Collide_Stream_PreRec_1, Collide_Stream_PreRec_2, Collide_Stream_PreRec_3, Collide_Stream_PreRec_4, Collide_Stream_PreRec_5, Collide_Stream_PreRec_6;
				Stream_t stream_ghost_dens_inlet, stream_ghost_dens_outlet;
				Stream_t stream_ReceivedDistr, stream_SwapOldAndNew;
				Stream_t stream_memCpy_CPU_GPU_domainEdge, stream_memCpy_GPU_CPU_domainEdge;
				Stream_t stream_Read_Data_GPU_Dens;
				Stream_t stability_check_stream;

#endif

#ifdef HEMELB_USE_GPU
				bool Initialise_GPU(iolets::BoundaryValues* iInletValues, iolets::BoundaryValues* iOutletValues, const util::UnitConverter* iUnits);	// Initialise the GPU - memory allocations

				bool initialise_GPU_WallShearStressMagn(iolets::BoundaryValues* iInletValues, iolets::BoundaryValues* iOutletValues, const util::UnitConverter* iUnits);	// Initialise the GPU - memory allocations
				bool initialise_GPU_LBGKSL();

				bool FinaliseGPU();
				bool Read_DistrFunctions_CPU_to_GPU(int64_t firstIndex, int64_t siteCount);

				bool Read_DistrFunctions_GPU_to_CPU_tot(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache);
				bool Read_DistrFunctions_GPU_to_CPU_FluidSites();

				bool Read_DistrFunctions_GPU_to_CPU_totalSharedFs();
				bool Read_DistrFunctions_CPU_to_GPU_totalSharedFs();

				bool Read_Macrovariables_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache);
				//bool Read_Macrovariables_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache, kernels::HydroVars<LB_KERNEL>& hydroVars(geometry::Site<geometry::LatticeData>&_site));

				std::vector<site_t> identify_Range_iolets_ID(site_t first_index, site_t upper_index,  int* n_local_Iolets, int* n_unique_local_Iolets);
				void count_Iolet_ID_frequency( std::vector<int> &vect , int Iolet_ID_index, int* frequency_ret);

				void read_WallMom_from_propertyCache(site_t firstIndex, site_t siteCount, const lb::MacroscopicPropertyCache& propertyCache, std::vector<util::Vector3D<double> >& wallMom_Iolet);
				void read_WallMom_correction_from_propertyCache(site_t firstIndex, site_t siteCount, const lb::MacroscopicPropertyCache& propertyCache, std::vector<double>& wallMom_correction_Iolet);

				bool memCpy_HtD_GPUmem_WallMom(site_t firstIndex, site_t siteCount, std::vector<util::Vector3D<double> >& wallMom_Iolet, void *GPUDataAddr_wallMom);
				bool memCpy_HtD_GPUmem_WallMom_correction(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_Iolet, void *GPUDataAddr_wallMom);
				bool memCpy_HtD_GPUmem_WallMom_correction_cudaStream(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_Iolet, void *GPUDataAddr_wallMom, Stream_t ptrStream);

				//IZ - April 2023
				bool memCpy_HtD_GPUmem_WallMom_prefactor_correction(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_prefactor_Iolet, void *GPUDataAddr_wallMom_prefactor);

				bool memCpy_HtD_GPUmem_Coords_Iolets(site_t firstIndex, site_t siteCount, void *GPUDataAddr_Coords_iolets);

				void get_Iolet_BCs(std::string& hemeLB_IoletBC_Inlet, std::string& hemeLB_IoletBC_Outlet);

				void swap_Pointers_GPU_glb_mem(void **pointer_GPU_glb_left, void **pointer_GPU_gbl_right);

				// Added December 2022 - Apply Boundary Conditions (GPU)
				void apply_Vel_BCs_File_GetWallMom_correction();

				// Added April 2023 - Vel BCs case
				void apply_Vel_BCs_File_GetWallMom_correction_ApprPref();
				void apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch();
				void apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreSend();
				void apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch_PreReceive();

				// Debugging Vel BCs case
				bool compare_CPU_GPU_WallMom_correction(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_Iolet, void *GPUDataAddr_wallMom);

				static void Cleanup_GPU_pinned_Memory();
#endif
//========================================================================
				//IZ

			private:

				void SetInitialConditions();

				void InitCollisions();
				// The following function pair simplify initialising the site ranges for each collider object.
				void InitInitParamsSiteRanges(kernels::InitParams& initParams, unsigned& state);
				void AdvanceInitParamsSiteRanges(kernels::InitParams& initParams, unsigned& state);
				/**
				 * Ensure that the BoundaryValues objects have all necessary fields populated.
				 */
				void PrepareBoundaryObjects();

				void ReadParameters();

				void handleIOError(int iError);

				// Collision objects
				tMidFluidCollision* mMidFluidCollision;
				tWallCollision* mWallCollision;
				tInletCollision* mInletCollision;
				tOutletCollision* mOutletCollision;
				tInletWallCollision* mInletWallCollision;
				tOutletWallCollision* mOutletWallCollision;

				template<typename Collision>
					void StreamAndCollide(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount)
					{
						collision->template StreamAndCollide<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache);
					}

				template<typename Collision>
					void PostStep(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount)
					{
						collision->template DoPostStep<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache);
					}

#ifdef HEMELB_USE_GPU
					// Added May 2020 - Useful for the GPU version
					template<typename Collision>
						void GetWallMom(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount, lb::MacroscopicPropertyCache& propertyCache)
						{
							collision->template GetWallMom<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache);
						}

					// Added March 2022 - Vel BCs case -Instead of wall momentum get the correction term
					template<typename Collision>
						void GetWallMom_correction(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount, lb::MacroscopicPropertyCache& propertyCache)
						{
							collision->template GetWallMom_correction<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache);
						}

					// Added July 2022 - Vel BCs case -Pass the single wall momentum correction term directly to the GPU
					template<typename Collision>
						//std::vector<distribn_t> GetWallMom_correction_Direct(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount, lb::MacroscopicPropertyCache& propertyCache)
						void GetWallMom_correction_Direct(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount, lb::MacroscopicPropertyCache& propertyCache, std::vector<double>& wallMom_correction_Iolet)
						{
							collision->template GetWallMom_correction_Direct<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache, wallMom_correction_Iolet);
						}

					// Added April 2023 - Vel BCs case
					//	Pass the prefactor (time-independent and geometry only dependent) associated with the single wall momentum correction term directly to the GPU at initialisation
						template<typename Collision>
							void GetWallMom_prefactor_correction_Direct(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount, lb::MacroscopicPropertyCache& propertyCache, std::vector<double>& wallMom_prefactor_correction_Iolet)
							{
								collision->template GetWallMom_prefactor_correction_Direct<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache, wallMom_prefactor_correction_Iolet);
							}

#endif

				unsigned int inletCount;
				unsigned int outletCount;

				configuration::SimConfig *mSimConfig;
				net::Net* mNet;
				geometry::LatticeData* mLatDat;
				SimulationState* mState;
				iolets::BoundaryValues *mInletValues, *mOutletValues;

				LbmParameters mParams;

				const util::UnitConverter* mUnits;

				hemelb::reporting::Timers &timings;

				MacroscopicPropertyCache propertyCache;

				geometry::neighbouring::NeighbouringDataManager *neighbouringDataManager;

		};

	} // Namespace lb
} // Namespace hemelb
#endif // HEMELB_LB_LB_H
