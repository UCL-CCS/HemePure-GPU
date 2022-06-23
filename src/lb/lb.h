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

				// Iolets Info: Used for the case of Pressure BCs (NASHZEROTHORDERPRESSUREIOLET)
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

				// And the corresponding host vectors related to the above
				std::vector<util::Vector3D<double> > wallMom_Inlet_Edge;
				std::vector<util::Vector3D<double> > wallMom_InletWall_Edge;
				std::vector<util::Vector3D<double> > wallMom_Inlet_Inner;
				std::vector<util::Vector3D<double> > wallMom_InletWall_Inner;
				std::vector<util::Vector3D<double> > wallMom_Outlet_Edge;
				std::vector<util::Vector3D<double> > wallMom_OutletWall_Edge;
				std::vector<util::Vector3D<double> > wallMom_Outlet_Inner;
				std::vector<util::Vector3D<double> > wallMom_OutletWall_Inner;

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

				// Cuda streams
				hipStream_t Collide_Stream_PreSend_1, Collide_Stream_PreSend_2, Collide_Stream_PreSend_3, Collide_Stream_PreSend_4, Collide_Stream_PreSend_5, Collide_Stream_PreSend_6;
				hipStream_t Collide_Stream_PreRec_1, Collide_Stream_PreRec_2, Collide_Stream_PreRec_3, Collide_Stream_PreRec_4, Collide_Stream_PreRec_5, Collide_Stream_PreRec_6;
				hipStream_t stream_ghost_dens_inlet, stream_ghost_dens_outlet;
				hipStream_t stream_ReceivedDistr, stream_SwapOldAndNew;
				hipStream_t stream_memCpy_CPU_GPU_domainEdge, stream_memCpy_GPU_CPU_domainEdge;
				hipStream_t stream_Read_Data_GPU_Dens;

#endif

#ifdef HEMELB_USE_GPU
				bool Initialise_GPU(iolets::BoundaryValues* iInletValues, iolets::BoundaryValues* iOutletValues, const util::UnitConverter* iUnits);	// Initialise the GPU - memory allocations

				bool Initialise_kernels_GPU(); // Initialise the kernels' setup

				bool FinaliseGPU();
				bool Read_DistrFunctions_CPU_to_GPU(int64_t firstIndex, int64_t siteCount);

				bool Read_DistrFunctions_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache);
				bool Read_DistrFunctions_GPU_to_CPU_totalSharedFs();
				bool Read_DistrFunctions_CPU_to_GPU_totalSharedFs();

				bool Read_Macrovariables_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache);
				//bool Read_Macrovariables_GPU_to_CPU(int64_t firstIndex, int64_t siteCount, lb::MacroscopicPropertyCache& propertyCache, kernels::HydroVars<LB_KERNEL>& hydroVars(geometry::Site<geometry::LatticeData>&_site));

				std::vector<site_t> identify_Range_iolets_ID(site_t first_index, site_t upper_index,  int* n_local_Iolets, int* n_unique_local_Iolets);
				void count_Iolet_ID_frequency( std::vector<int> &vect , int Iolet_ID_index, int* frequency_ret);

				void read_WallMom_from_propertyCache(site_t firstIndex, site_t siteCount, lb::MacroscopicPropertyCache& propertyCache, std::vector<util::Vector3D<double> >& wallMom_Iolet);
				bool memCpy_HtD_GPUmem_WallMom(site_t firstIndex, site_t siteCount, std::vector<util::Vector3D<double> >& wallMom_Iolet, void *GPUDataAddr_wallMom);
		  
				void get_Iolet_BCs(std::string hemeLB_IoletBC_Inlet, std::string hemeLB_IoletBC_Outlet);
		  
		  		void swap_Pointers_GPU_glb_mem(void **pointer_GPU_glb_left, void **pointer_GPU_gbl_right);
		  
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
						void GetWallMom(Collision* collision, const site_t iFirstIndex, const site_t iSiteCount)
						{
							collision->template GetWallMom<false> (iFirstIndex, iSiteCount, &mParams, mLatDat, propertyCache);
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
