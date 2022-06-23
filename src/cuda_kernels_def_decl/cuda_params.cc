// This file is part of the GPU development for HemeLB
// 7-1-2019

/**
================================================================================
	HemeLB version 1.13

	24 March 2020:
		First GPU version of HemeLB, which uses the following options:
		1. Collision kernel: BGK - single relaxation time approximation.
		2. Wall BCs: Simple Bounce Back
				this is the first out of 4 possible options : the other ones are BFL, GZS, JUNKYANG).
		3. Inle/Outlet BCs: NashZerothOrderPressure
				this is the first out of 2 possible options - the other one is Ladd's BCs).

	1 April 2020:
		a. 	Changed all the collision-streaming kernels to be the same in PreSend and PreReceive
		b. 	Use neighbouring index that contains the ACTUAL streaming ADDRESS in global memory for f_new - NOT the fluid ID

	5 April 2020:
		a. Abort simulation if Initialising the GPU fails
				function bool LBM<LatticeType>::Initialise_GPU(iolets::BoundaryValues* iInletValues, iolets::BoundaryValues* iOutletValues)
		b. Change the position of Initialise_GPU
					From inside void LBM<LatticeType>::Initialise(...)
					to be called directly from SimulationMaster
//------------------------------------------------------------------------------

	HemeLB version 1.15
	12	April 2020
		1. 	GPU kernel for calculating MacroVariables: Density and Velocity
							placed at the BeginPhase step (function RequestComms in lb.hpp)
								bool IteratedAction::CallAction(int action)
		    				{
					      switch (static_cast<phased::steps::Step>(action))
					      {
					        case phased::steps::BeginPhase:
					          RequestComms();
							slows down the performance though...
			****		Keep the option of calculating density and momentum	****
			****				during the collision kernels.										****

		2. Merge the collision-streaming kernels Type 1 & 2: midDomain and Walls
					GPU kernel: GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB

		3. Save the MacroVariables (density and velocity) to the GPU global memory from
				the GPU collision-streaming kernels every 100 time-steps.
				Needs the time of the LB iteration to be passed to the collision-streaming kernels
				Hence, reduces the frequency for writing to GPU global memory

		4. Swap the post-collision populations:
					Compare the GPU kernel for SwapOldAndNew (GPU_SwapOldAndNew) to the memory
					copy cudaMemcpyDeviceToDevice.
					The later (cudaMemcpyDeviceToDevice) seems to be better. Keep this...
				*** KEEP the cudaMemcpyDeviceToDevice option ***

	21 April 2020
		1. Case of multiple iolets:
		 			Function returns the local iolets and the fluid sites associated with each iolet
						returned vector - copied to device_vector (thrust) and then passed
						to the GPU kernels once converted back to its underlying pointer (using thrust::raw_pointer_cast)
//------------------------------------------------------------------------------

  HemeLB version 1.16
	23 April 2020
		1. Case of multiple iolets:
					a. Replace the thrust device vector with arrays in GPU constant memory
							Different versions of the iolets Kernels call different gpu constant memory arrays
							Template the above.
//------------------------------------------------------------------------------

  HemeLB version 1.17
	27 April 2020
		1. Case of multiple iolets:
				Pass a struct to the GPU kernels (iolets and iolets with walls) containing the relevant info.

		2. Set the current GPU device in SimulationMaster.cu (function check_GPU_capabilities)
//------------------------------------------------------------------------------

	HemeLB version 1.18
	1 May 2020
		1. Case of multiple iolets:
		 		Error in handling the iolet IDs in case of irregular indexing of iolets in a rank
				Complete method to identify iolet ID and fluid sites range [min_index, max_index]
					function identify_Range_iolets_ID
						returns:
							a. the vector with this information
							b. the number of local iolets by increasing Fluid ID (it is possible to have irregular indexing, for example. boundary_Iolet_ID: 0 1 2 1 2 3 2 )
							c. the unique number of local iolets, which in the example above is 4: boundary_Iolet_ID: 0 1 2 3
//------------------------------------------------------------------------------

HemeLB version 1.19
1 May 2020
	1. Clean up function Initialise_GPU() to minimise the memory allocated - Done!!!

	2. Change the order of things: Done!!! DIDN'T WORK AS EXPECTED - SLOWER!!!
						a. In PostReceive:
								a.1. Synchronisation barrier for the PreReceive collision streaming kernels (complete the streaming in fNew_GPU_b)
								a.2. Do the swap of distr. functions fNew_GPU_b[0, nFluidnodes) placed into fOld_GPU_b[0, nFluidnodes)
						b. Do the streaming of the received distr. functions (placed in fOld_GPU_b in totalSharedFs) but instead of streaming these in f_new
								place these in f_old
						These should overlap the 2 above processes and will also close any time gaps after PreReceive.

	The above did not work as expected... Slower code overall, tested with the bifurcation_hires case

	3. Also tried reverting to the original steps in hemeLB (see what is in version 1.20) - BUt was slower.

		Revert everything back to how it was, i.e.

					#ifdef HEMELB_USE_GPU
							// Order significant here
							// BeginPhase must begin and EndPhase must end, those steps which should be called for a given phase.
							BeginAll = -1, // Called only before first phase
							BeginPhase = 0,
							Receive = 1,
							PreSend = 2,
							PreWait = 3,
							Send = 4,
							Wait = 5,
							EndPhase = 6,
							EndAll = 7, // Called only after final phase
					#else
							// Order significant here
							// BeginPhase must begin and EndPhase must end, those steps which should be called for a given phase.
							BeginAll = -1, // Called only before first phase
							BeginPhase = 0,
							Receive = 1,
							PreSend = 2,
							Send = 3,
							PreWait = 4,
							Wait = 5,
							EndPhase = 6,
							EndAll = 7, // Called only after final phase
					#endif
//------------------------------------------------------------------------------

	HemeLB version 1.19.a
	Same as version 1.19 - Clean-up unecessary printf statements - USE THIS VERSION FOR TESTING/SCALING
//------------------------------------------------------------------------------

//
VERSIONS v*.a will keep the modified sequence of steps
 while
VERSIONS v*.b will follow the original sequence
//

//------------------------------------------------------------------------------
	HemeLB version 1.20.a
	14 May 2020

	1. Shift the Synchronisation barrier in PreReceive() for the collision-streaming kernels in PreSend()
			which is necessary for the exchange of data at domain boundaries

//------------------------------------------------------------------------------
	HemeLB version 1.21.a
	15 May 2020

	1. Implement Velocity BCs at iolets - LaddIolet iolet BCs - Done!!!

	1.1.  Read the wallMom variable at each one of the inlet/outlet sites following the same delegate scheme as in HemeLB
				*** Important remark: We do not apply the correction (multiplication by the local density) on the wall mom.
				***   								Hence, we have to apply this, when the values are passed to the GPU. (see the relevant collision-streaming kernels for more details)

	 			Note the following on the implementation:
					CMake file determines whether:
						a. Pressure (NASHZEROTHORDERPRESSUREIOLET)
						b. Velocity BCs (LADDIOLET)
					Input file: Specifies which one of the 3 different options for LaddIolet BCs to implement:
						1. parabolic
						2. From file
						3. Wommersley
	1.2. Save the wallmom in propertyCache (wallMom_Cache): function GetWallMom (in lb.hpp)
	1.3. Read from propertyCache: function read_WallMom_from_propertyCache  (in lb.hpp)
	1.4. Allocate memory on the GPU and perform a memcpy (HtD) for wallMom: function memCpy_HtD_GPUmem_WallMom (in lb.hpp)
	 			for each iolet type:
					i.e. Inlet, Inlet with walls, Outlet, Outlet with walls and
								for the PreSend and PreReceive steps
					resulting up to 8 different possible arrays and the following pointers to thet data in GPU global memory:
							void *GPUDataAddr_wallMom_Inlet_Edge;
							void *GPUDataAddr_wallMom_InletWall_Edge;
							void *GPUDataAddr_wallMom_Inlet_Inner;
							void *GPUDataAddr_wallMom_InletWall_Inner;
							void *GPUDataAddr_wallMom_Outlet_Edge;
							void *GPUDataAddr_wallMom_OutletWall_Edge;
							void *GPUDataAddr_wallMom_Outlet_Inner;
							void *GPUDataAddr_wallMom_OutletWall_Inner;
	1.5. Within the kernels (GPU_CollideStream_Iolets_Ladd_VelBCs & GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs)
				Multiply with the local density (correction if CollisionType::CKernel::LatticeType::IsLatticeCompressible() == true)
	 			TODO: Pass the boolean Variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()


//------------------------------------------------------------------------------
	HemeLB version 1.22.a
	16 June 2020

	1. 	Modify the Pressure BCs case (Iolets) and remove the struct passed to the kernels
				containing the information for the iolet (NOT Possible to have the FAM within the context of C++)
			ACTION:
				TODO: Pass this info to the GPU global memory at Initialise_GPU instead. Done!!!
							Modify the collision-streaming kernels accordingly:
								GPU_CollideStream_Iolets_NashZerothOrderPressure_v2

	2. 	TODO:
				Switch to asynchronous memcpy H2D for the wall momentum and the case of Velocity Inlet/Outlet BCs
					in function:
						bool LBM<LatticeType>::memCpy_HtD_GPUmem_WallMom

	July 2020 - New approach
	Switch between 2 versions of the collision-streaming kernels for the Iolets collision-streaming kernels (Types 3-6)
		Only difference on how the iolets info is passed to the GPU (iolets_ID_range)
			depending on local number of iolets (less than limit - pass the struct, otherwise pointer to data in global memory)

		Option (a): Pass a struct when number of local iolets is less than local_iolets_MaxSIZE/3
									See in cuda_params.h:  #define local_iolets_MaxSIZE 90
		Option (b): Pass this info to gpu global memory, when number of local iolets is greater than above.


//------------------------------------------------------------------------------
	HemeLB version 1.23.a

	16 July 2020
	TODO: 1. Asynchronous writing to disk. Need to understand how writing is done!!!

				2. Change the sequence of launching the Collision-streaming kernels (Types 3-6 first and then 1-2 and test the impact)

	August 2020
	1. Asynchronous writing to disk. Done!!!
			Notes:
				Use MPI I/O - MPI I/0 currently using MPI_File_write_at
				Switch to non-blocking, using MPI_File_iwrite_at and MPI_Wait

//------------------------------------------------------------------------------
	HemeLB version 1.24.a

	Sept 2020
	TODO: 1. CUDA-aware mpi
							Flag: HEMELB_CUDA_AWARE_MPI
						Make the changes in
							void BaseNet::Receive()
							void BaseNet::Send()
							
				2. Pointers to GPU global memory must be transfered from class LBM to class geometry::LatticeData
				 				class LBM is friend class of class LatticeData
								hence object of class LBM can access the private and protected members of the class geometry::LatticeData


	##############################################################################
	To do - Think about the following:
	1. 	Use 256 threads per block for the Collision type 1
											Use 128/64 for the other types of Collision

	2. 	Change the way of allocating memory for:
				I. Data_uint32_IoletIntersect (GPUDataAddr_uint32_Iolet)
				II. Data_uint32_WallIntersect (GPUDataAddr_uint32_Wall)
			as I currently define the above variables for each fluid node - Too much!!!
			It should be done only for the corresponding number of fluid nodes involved in these collision-streaming types. Check the limits from there.

	3. 	Transfer in Initialise_kernels_GPU all the set-up for the individual cuda kernels
				No need to repeat at each time-step the configuration steps

	4. Evaluate other macrovariables on the GPU - memcpy to host according to specified frequency for saving to disk

	5. Stability evaluation every X timesteps on the GPU - maybe at the same time when evaluating/saving MacroVariables
	    Maybe add this when saving the velocity and density at each one of the collsiion-streaming kernels

	6. Pass the boolean Variable:
				CollisionType::CKernel::LatticeType::IsLatticeCompressible()
			to the iolets collision-streaming kernels associated with Vel BCs (LADDIOLET)
	##############################################################################

	//----------------------------------------------------------------------------

================================================================================
*/



#include <hip/hip_runtime.h>
#include <stdio.h>



#ifdef HEMELB_USE_GPU
#include "cuda_kernels_def_decl/cuda_params.h"

//#include <thrust/host_vector.h>
//#include <thrust/device_vector.h>
//#include <thrust/copy.h>
//#include <thrust/fill.h>
#endif


namespace hemelb
{

#ifdef HEMELB_USE_GPU

	// GPU constant memory
	 /*__constant__ site_t _Iolets_Inlet_Edge[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_InletWall_Edge[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_Inlet_Inner[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_InletWall_Inner[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_Outlet_Edge[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_OutletWall_Edge[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_Outlet_Inner[local_iolets_MaxSIZE];
	 __constant__ site_t _Iolets_OutletWall_Inner[local_iolets_MaxSIZE];


	__constant__ unsigned int _NUMVECTORS;
	__constant__ double dev_tau;
	__constant__ double dev_minusInvTau;
	__constant__ double _Cs2;

	__constant__ int _InvDirections_19[19];

	__constant__ double _EQMWEIGHTS_19[19];

	__constant__ int _CX_19[19];
	__constant__ int _CY_19[19];
	__constant__ int _CZ_19[19];

	__constant__ int _WriteStep = 1000;
	__constant__ int _Send_MacroVars_DtH = 1;*/


	//===================================================================================================================
	//setter for const device memory. Have to be in the same compilation unit due to clang's bug: https://reviews.llvm.org/D95901
	
	void d95901_set_numvectors(const unsigned int numvectors, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_NUMVECTORS), &numvectors, sizeof(numvectors), 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_EQMWEIGHTS_19(const double* eqmweights, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_EQMWEIGHTS_19), eqmweights, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_InvDirections_19(const hemelb::Direction* invdir, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_InvDirections_19), invdir, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_CX_19(const int* cx, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_CX_19), cx, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_CY_19(const int* cy, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_CY_19), cy, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_CZ_19(const int* cz, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_CZ_19), cz, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_dev_tau(const double* devtau, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::dev_tau), devtau, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_dev_minusInvTau(const double* devminusInvTau, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::dev_minusInvTau), devminusInvTau, size, 0, hipMemcpyHostToDevice);
	}
	
	void d95901_set_Cs2(const double* cs2, size_t size, hipError_t* status){
		*status = hipMemcpyToSymbol(HIP_SYMBOL(hemelb::_Cs2), cs2, size, 0, hipMemcpyHostToDevice);
	}
	
	//===================================================================================================================

	/**
	__global__ GPU kernels
	*/

	//**************************************************************
	// Kernel for the Collision step
	// for the Lattice Boltzmann algorithm
	// Merged Collision Types 1 & 2:
	// 		Collision Type 1: Mid Domain - All neighbours are Fluid nodes
	// 		Collision Type 2: mWallCollision: Wall-Fluid interaction
	//	Fluid sites range: [lower_limit_MidFluid, upper_limit_Wall)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the wall-fluid links - Done!!!
	//**************************************************************
	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB(distribn_t* GMem_dbl_fOld_b,
										distribn_t* GMem_dbl_fNew_b,
										distribn_t* GMem_dbl_MacroVars,
										site_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										site_t nArr_dbl,
										site_t lower_limit_MidFluid, site_t upper_limit_MidFluid,
										site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, int time_Step)
	{
	  unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
	  
	  Ind = Ind + lower_limit_MidFluid;
	  
	  if(Ind >= upper_limit_Wall)
	    return;
	  
	  //printf("lower_limit_MidFluid: %lld, upper_limit_MidFluid: %lld, lower_limit_Wall: %lld, upper_limit_Wall: %lld \n\n", lower_limit_MidFluid, upper_limit_MidFluid, lower_limit_Wall, upper_limit_Wall);





		// Load the distribution functions

		//f[19] and fEq[19]

		double dev_ff[19];
		double nn = 0.0;	// density
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;
		
		double velx, vely, velz;	// Fluid Velocity



		//-----------------------------------------------------------------------------------------------------------

		// 1. Read the fOld_GPU_b distr. functions

		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions

		// 		a. Calculate density

		// 		b. Calculate momentum - Note: No body forces

		/*for(int direction = 0; direction< _NUMVECTORS; direction++){

			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];

		}



		for(int direction = 0; direction< _NUMVECTORS; direction++){

			momentum_x += (double)_CX_19[direction] * dev_ff[direction];

			momentum_y += (double)_CY_19[direction] * dev_ff[direction];

			momentum_z += (double)_CZ_19[direction] * dev_ff[direction];

		}

		*/

#pragma unroll 19
		for(int direction = 0; direction< _NUMVECTORS; direction++){
		  double ff = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];
		  dev_ff[direction] = ff;
		  nn += ff;
		  
		  // Shows a lower number of registers per thread (51) compared to the the explicit method below!!!
		  momentum_x += (double)_CX_19[direction] * ff;
		  momentum_y += (double)_CY_19[direction] * ff;
		  momentum_z += (double)_CZ_19[direction] * ff;
		}

		/*
		// Evaluate momentum explicitly - The number of registers per thread increases though (56 from 51) compared to the approach of multiplying with the lattice direction's projections !!! Why?
		// Based on HemeLB's vector definition
		momentum_x = dev_ff[1] - dev_ff[2] + dev_ff[7]  - dev_ff[8]  + dev_ff[9]  - dev_ff[10] + dev_ff[11] - dev_ff[12] + dev_ff[13] - dev_ff[14]; // HemeLB vector definition is different than the one I am using
		momentum_y = dev_ff[3] - dev_ff[4] + dev_ff[7]  - dev_ff[8]  - dev_ff[9]  + dev_ff[10] + dev_ff[15] - dev_ff[16] + dev_ff[17] - dev_ff[18];
		momentum_z = dev_ff[5] - dev_ff[6] + dev_ff[11] - dev_ff[12] - dev_ff[13] + dev_ff[14] + dev_ff[15] - dev_ff[16] - dev_ff[17] + dev_ff[18];
		//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
		*/
		
		double density_1 = 1.0 / nn;
		
		// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;;
		velz = momentum_z * density_1;;
		
		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions

		double momentumMagnitudeSquared = momentum_x * momentum_x
		  + momentum_y * momentum_y + momentum_z * momentum_z;
		
		site_t index_wall = nArr_dbl * _NUMVECTORS; // typedef int64_t site_t;
		
#pragma unroll 19
		for (int i = 0; i < _NUMVECTORS; ++i)
		  {
		    double mom_dot_ei = (double)_CX_19[i] * momentum_x
		      + (double)_CY_19[i] * momentum_y
		      + (double)_CZ_19[i] * momentum_z;
		    
		    double dev_fEq = _EQMWEIGHTS_19[i]
		      * (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
			 + (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		    
		    //double ff = dev_ff[i] + (dev_ff[i] - dev_fEq) * dev_minusInvTau;
		    double ff = dev_ff[i] * (1 + dev_minusInvTau) - dev_fEq * dev_minusInvTau;
		    
		    //-----------------------------------------------------------------------------------------------------------
		    // d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		    //-----------------------------------------------------------------------------------------------------------
		    
		    // Collision step:
		    // Single Relaxation Time approximation (LBGK)
		    //double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements
		    
		    // Evolution equation for the fi's here
		    //for (int i = 0; i < _NUMVECTORS; ++i)
		    //{
		    //	dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau;
		    //}
		    

		    // --------------------------------------------------------------------------------
		    // Streaming Step:
		    // a. Load the streaming indices
		    // b. If within the limits for the mWallCollision
		    //		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes
		    //		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)
		    
		    //site_t index_wall = nArr_dbl * _NUMVECTORS; // typedef int64_t site_t;
		    
		    //for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
		    int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)i * nArr_dbl + Ind]; // Neighbouring index refers to the index to be streamed to in the global memory. Here it Refers to Data Address NOT THE STREAMING FLUID ID!!!

		    // Is there a performance gain in choosing Option 1 over Option 2 or Option 3 below???
		    // Option 1:
		    if(dev_NeighInd == index_wall) // Wall Link
		      {
			// Simple Bounce Back case:
			GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[i] * nArr_dbl + Ind]= ff; // Bounce Back - Same fluid ID - Reverse LB_Dir
		      }
		    else{
		      GMem_dbl_fNew_b[dev_NeighInd] = ff; 	// If neigh_d is selected
		    }
		    //printf("Local ID : %llu, Mem. Location: %.llu, LB_dir = %d, Neighbour = %llu \n\n", Ind, local_fluid_site_mem_loc, LB_Dir, dev_NeighInd);

		    
		    /*
		      
		    // Option 2: Use of ternary operator to replace the if-else statement
		    
		    int64_t arr_index = (dev_NeighInd == index_wall) ? (unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind : dev_NeighInd;
		    
		    GMem_dbl_fNew_b[arr_index] = dev_ff[LB_Dir];
		    
		    */
		    
		    
		    
		    /*
		      
		    // Option 3: Avoid the if-else operator by multiplying with a boolean variable (wall link or not)
		    
		    bool is_Wall_link_test = (dev_NeighInd == index_wall);
		    
		    int64_t arr_index = ((unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind) * is_Wall_link_test + dev_NeighInd * (!is_Wall_link_test);
		    
		    GMem_dbl_fNew_b[arr_index] = dev_ff[LB_Dir];
		    
		    */

		}





		// --------------------------------------------------------------------------------

		// Streaming Step:

		// a. Load the streaming indices



		// b. If within the limits for the mWallCollision

		//		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes

		//		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)



		/*

		if ( (Ind - upper_limit_MidFluid +1)*(Ind - lower_limit_MidFluid) <= 0){

		//if( (Ind >= lower_limit_MidFluid) && ( Ind < upper_limit_MidFluid) ){

			for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

					// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)

					// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

					int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];



					// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well

					// fNew populations:

					// GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd] = dev_ff[LB_Dir]; // If neigh_c is selected and dev_NeighInd contains the fluid_ID info



					GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected

					// GMem_dbl_fNew_b[dev_NeighInd[LB_Dir]] = dev_ff[LB_Dir]; 	// If neigh_d is selected

				}

		}

		else

		{

			for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

					int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Neighbouring index refers to the index to be streamed to in the global memory. Here it Refers to Data Address NOT THE STREAMING FLUID ID!!!



					if(dev_NeighInd == (nArr_dbl * _NUMVECTORS)) // Wall Link

					{

						// Simple Bounce Back case:

						GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir

					}

					else{

						GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected

					}



			}

		}

		*/

		//=============================================================================================

		// Write old density and velocity to memory -

		if (time_Step%_Send_MacroVars_DtH ==0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}


	} // Ends the merged kernels GPU_Collide Types 1 & 2: mMidFluidCollision & mWallCollision
	//==========================================================================================








	//**************************************************************
	// Kernel for the Collision step
	// for the Lattice Boltzmann algorithm
	// Collision Type 2: mWallCollision: Wall-Fluid interaction
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the wall-fluid links - Done!!!
	//**************************************************************
	__global__ void GPU_CollideStream_mWallCollision_sBB(double* GMem_dbl_fOld_b,
										double* GMem_dbl_fNew_b,
										double* GMem_dbl_MacroVars,
										int64_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										uint64_t nArr_dbl,
										uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19];
		double nn = 0.0;	// density
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;

		double velx, vely, velz;	// Fluid Velocity

		//-----------------------------------------------------------------------------------------------------------
		// 1. Read the fOld_GPU_b distr. functions
		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
		// 		a. Calculate density
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)_CX_19[direction] * dev_ff[direction];
			momentum_y += (double)_CY_19[direction] * dev_ff[direction];
			momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
			//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
		}

		/*
		// In the case of body force
		momentum_x += 0.5 * _force_x;
		momentum_y += 0.5 * _force_y;
		momentum_z += 0.5 * _force_z;
		*/

		double density_1 = 1.0 / nn;

		// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;;
		velz = momentum_z * density_1;;

		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}

		//__syncthreads(); // Check if needed!


		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices
		// b. The Wall-Fluid links info

		//
		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		// a. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		//int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!

			// Put the new populations after collision in the GMem_dbl array,
			// implementing the streaming step with Simple Bounce Back if Wall-Fluid link

			// fNew (dev_fn) populations:
			// Check is direction LB_Dir is a wall link
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask);

			if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//printf("Site ID = %lld - Wall in Dir: %d \n\n", Ind, LB_Dir);
				// Simple Bounce Back case:
				GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir
			}
			else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//---------------------------------------------------------------------------
				// If it streams in direction inside the simulation domain then it will point to a fluid ID < nFluid_nodes, otherwise it will stream to a neighbouring rank (place in the totalSharedFs at the end of the array)
				if (dev_NeighInd < (nArr_dbl*_NUMVECTORS) ) // maximum Data Address in array that correspond to this domain = nFluid_nodes*_NUMVECTORS
				{
					dev_NeighInd = (dev_NeighInd - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index

					// Save the post collision population in fNew
					GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd] = dev_ff[LB_Dir];
				}
				else{
					// Save the post collision population in fNew[Addr] at the end of the array in the (1+totalSharedFs) location
					GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir];
					//
					// Debugging - Remove later
					// Check if it points to an address outside the (nFluid_nodes * _NUMVECTORS + 1+totalSharedFs )
					//if (dev_NeighInd >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, dev_NeighInd[LB_Dir] );
					//
				}
				//---------------------------------------------------------------------------
			} // Closes the bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

		} // Closes the loop over the LB_dir

		//=============================================================================================
		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		GMem_dbl_MacroVars[Ind] = nn;
		GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
		GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
		GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

	} // Ends the kernel GPU_Collide Type 2: mWallCollision: Case Fluid-Wall collision
	//==========================================================================================


	//**************************************************************
	// Kernel for the Collision step
	// for the Lattice Boltzmann algorithm
	// Collision Type 2: mWallCollision: Wall-Fluid interaction
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the wall-fluid links - Done!!!
	//**************************************************************
	__global__ void GPU_CollideStream_mWallCollision_sBB_PreRec(double* GMem_dbl_fOld_b,
										double* GMem_dbl_fNew_b,
										double* GMem_dbl_MacroVars,
										int64_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										uint64_t nArr_dbl,
										uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19];
		double nn = 0.0;	// density
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;

		double velx, vely, velz;	// Fluid Velocity

		//-----------------------------------------------------------------------------------------------------------
		// 1. Read the fOld_GPU_b distr. functions
		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
		// 		a. Calculate density
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)_CX_19[direction] * dev_ff[direction];
			momentum_y += (double)_CY_19[direction] * dev_ff[direction];
			momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
			//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
		}

		/*
		// In the case of body force
		momentum_x += 0.5 * _force_x;
		momentum_y += 0.5 * _force_y;
		momentum_z += 0.5 * _force_z;
		*/

		double density_1 = 1.0 / nn;

		// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;;
		velz = momentum_z * density_1;;

		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}

		// __syncthreads(); // Check if needed!


		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices
		// b. The Wall-Fluid links info

		site_t index_wall = nArr_dbl * _NUMVECTORS;
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Neighbouring index refers to the index to be streamed to in the global memory. Here it Refers to Data Address NOT THE STREAMING FLUID ID!!!

				if(dev_NeighInd == index_wall) // Wall Link
				{
					// Simple Bounce Back case:
					GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir
				}
				else{
					GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected
				}
		}

		/*
		//
		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		//__syncthreads();

		// a. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		//int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){


			// Put the new populations after collision in the GMem_dbl array,
			// implementing the streaming step with Simple Bounce Back if Wall-Fluid link

			// fNew (dev_fn) populations:
			// Check is direction LB_Dir is a wall link
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask);

			if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//printf("Site ID = %lld - Wall in Dir: %d \n\n", Ind, LB_Dir);
				// Simple Bounce Back case:
				GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir
			}
			else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//---------------------------------------------------------------------------
				// If we use the elements in GMem_int64_Neigh_d - then we access the memory address in fOld or fNew directly (Method B: data arranged by LB_Dir)..
				// Including the info for the totalSharedFs (propagate outside of the simulation domain).
				// (remember the memory layout in hemeLB is based on the site fluid index (Method A), i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Depends on which neigh array is loaded... Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!

				// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well
				// fNew populations:
				// GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd] = dev_ff[LB_Dir]; // If neigh_c is selected
				GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected

				//---------------------------------------------------------------------------
			} // Closes the bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

		} // Closes the loop over the LB_dir
		*/

		//=============================================================================================
		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%_Send_MacroVars_DtH==0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 2: mWallCollision: Case Fluid-Wall collision
	//==========================================================================================




	//===================================================================================================================
	//**************************************************************
	// Kernel for the Collision step
	// for the Lattice Boltzmann algorithm
	// Collision Type 1: Mid Domain - All neighbours are Fluid nodes
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	//**************************************************************
	__global__ void GPU_CollideStream_1_PreReceive(	double* GMem_dbl_fOld_b,
													double* GMem_dbl_fNew_b,
													double* GMem_dbl_MacroVars,
													int64_t* GMem_int64_Neigh,
													uint64_t nArr_dbl,
													uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;


		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19];
		double nn = 0.0;	// density
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;

		double velx, vely, velz;	// Fluid Velocity

		//-----------------------------------------------------------------------------------------------------------
		// 1. Read the fOld_GPU_b distr. functions
		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
		// 		a. Calculate density
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)_CX_19[direction] * dev_ff[direction];
			momentum_y += (double)_CY_19[direction] * dev_ff[direction];
			momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
			//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
		}

		/*
		// In the case of body force
		momentum_x += 0.5 * _force_x;
		momentum_y += 0.5 * _force_y;
		momentum_z += 0.5 * _force_z;
		*/
		double density_1 = 1.0 / nn;

		// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;
		velz = momentum_z * density_1;

		//-----------------------------------------------------------------------------------------------------------
	  // c. Calculate equilibrium distr. functions		double momentumMagnitudeSquared = momentum_x * momentum_x
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		}
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
  	for (int i = 0; i < _NUMVECTORS; ++i)
  	{
    	//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
    	dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
  	}

		//__syncthreads(); // Check if needed!


		// Streaming Step -Load the streaming indices
		// dev_NeighInd[19] refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		//int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		// To increase performance just define this as int64_t

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
			//dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!

			// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well
			// fn populations:

			// If it streams in direction inside the simulation domain then it will point to a fluid ID < nFluid_nodes, otherwise it will stream to a neighbouring rank (place in the totalSharedFs at the end of the array)
			if (dev_NeighInd < (nArr_dbl*_NUMVECTORS) ) // maximum Data Address in array that correspond to this domain = nFluid_nodes*_NUMVECTORS
			{
				dev_NeighInd = (dev_NeighInd - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index

				// Save the post collision population in fNew
				GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd] = dev_ff[LB_Dir];

				//GMem_dbl_fNew_b[(unsigned long long)i * nArr_dbl + Ind] = dev_ff[i];	// No streaming - Just saves the post collision distribution value
			}
			else{
				// Save the post collision population in fNew[Addr] at the end of the array in the (1+totalSharedFs) location
				GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir];

				// Check if it points to an address outside the (nFluid_nodes * _NUMVECTORS + 1+totalSharedFs )
				//if (dev_NeighInd >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, dev_NeighInd[LB_Dir] );
			}
		}

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		GMem_dbl_MacroVars[Ind] = nn;
		GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
		GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
		GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

	} // Ends the kernel GPU_Collide
	//==========================================================================================



	//**************************************************************
	// Kernel for the Collision step
	// for the Lattice Boltzmann algorithm
	// Collision Type 1: Mid Domain - All neighbours are Fluid nodes
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	//
	// 	Reads the MacroVariables: the density and momentum (velocity) from GPU global memory
	//**************************************************************
	__global__ void GPU_CollideStream_1_PreReceive_noSave(	double* GMem_dbl_fOld_b,
													double* GMem_dbl_fNew_b,
													double* GMem_dbl_MacroVars,
													int64_t* GMem_int64_Neigh,
													uint64_t nArr_dbl,
													uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

		//-----------------------------------------------------------------------------------------------------------
		//Density
		distribn_t nn = GMem_dbl_MacroVars[Ind];
		// Fluid Velocity;
		distribn_t velx, vely, velz;
		velx = GMem_dbl_MacroVars[1ULL *nArr_dbl + Ind];
		vely = GMem_dbl_MacroVars[2ULL *nArr_dbl + Ind];
		velz = GMem_dbl_MacroVars[3ULL *nArr_dbl + Ind];

		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19];

		//-----------------------------------------------------------------------------------------------------------
		// Read the fOld_GPU_b distr. functions
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];
		}

		// Momentum
		double momentum_x, momentum_y, momentum_z;
		momentum_x = velx * nn;
		momentum_y = vely * nn;
		momentum_z = velz * nn;

		/*
		// In the case of body force
		momentum_x += 0.5 * _force_x;
		momentum_y += 0.5 * _force_y;
		momentum_z += 0.5 * _force_z;
		*/
		//-----------------------------------------------------------------------------------------------------------
		double density_1 = 1.0 / nn;

		//-----------------------------------------------------------------------------------------------------------
		// Calculate equilibrium distr. functions		double momentumMagnitudeSquared = momentum_x * momentum_x
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		}
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}
		//__syncthreads(); // Check if needed!


		// Streaming Step -Load the streaming indices
		/*// dev_NeighInd[19] refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		// 	To increase performance just define this as int64_t
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];
		}
		*/

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			// Streams definitely within the simulation domain, i.e. the streaming address is within (nFluid_nodes*_NUMVECTORS)
			// Hence use the Neighbouring Index given in GPUDataAddr_int64_Neigh_c, which is the actual streaming Fluid Index

			int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];

			// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well
			// fNew populations:
			// GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd] = dev_ff[LB_Dir]; // If neigh_c is selected and dev_NeighInd contains the fluid_ID info

			GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected
			// GMem_dbl_fNew_b[dev_NeighInd[LB_Dir]] = dev_ff[LB_Dir]; 	// If neigh_d is selected

		}

	} // Ends the kernel GPU_Collide
	//==========================================================================================



	//===================================================================================================================

	//**************************************************************
	// Kernel for the Collision step
	// for the Lattice Boltzmann algorithm
	// Collision Type 1: Mid Domain - All neighbours are Fluid nodes
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	//
	// 	Calculates MacroVariables: the density and momentum (velocity) from the distr. functions
	// 	Saves the MacroVariables
	//**************************************************************
	__global__ void GPU_CollideStream_1_PreReceive_SaveMacroVars(	double* GMem_dbl_fOld_b,
													double* GMem_dbl_fNew_b,
													double* GMem_dbl_MacroVars,
													int64_t* GMem_int64_Neigh,
													uint64_t nArr_dbl,
													uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;


		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19];
		double nn = 0.0;	// density
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;

		double velx, vely, velz;	// Fluid Velocity

		//-----------------------------------------------------------------------------------------------------------
		// 1. Read the fOld_GPU_b distr. functions
		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
		// 		a. Calculate density
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)_CX_19[direction] * dev_ff[direction];
			momentum_y += (double)_CY_19[direction] * dev_ff[direction];
			momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
			//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
		}

		/*
		// In the case of body force
		momentum_x += 0.5 * _force_x;
		momentum_y += 0.5 * _force_y;
		momentum_z += 0.5 * _force_z;
		*/
		double density_1 = 1.0 / nn;

		// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;
		velz = momentum_z * density_1;

		//-----------------------------------------------------------------------------------------------------------
	  // c. Calculate equilibrium distr. functions		double momentumMagnitudeSquared = momentum_x * momentum_x
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		}
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
  	for (int i = 0; i < _NUMVECTORS; ++i)
  	{
    	//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
    	dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
  	}
		//__syncthreads(); // Check if needed!


		// Streaming Step -Load the streaming indices
		/*// dev_NeighInd[19] refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		// 	To increase performance just define this as int64_t
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];
		}
		*/

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			// Streams definitely within the simulation domain, i.e. the streaming address is within (nFluid_nodes*_NUMVECTORS)
			// Hence use the Neighbouring Index given in GPUDataAddr_int64_Neigh_c, which is the actual streaming Fluid Index

			int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];

			// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well
			// fNew populations:
			// GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd] = dev_ff[LB_Dir]; // If neigh_c is selected and dev_NeighInd contains the fluid_ID info

			GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected
			// GMem_dbl_fNew_b[dev_NeighInd[LB_Dir]] = dev_ff[LB_Dir]; 	// If neigh_d is selected

		}


		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%_Send_MacroVars_DtH==0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide
	//==========================================================================================



	//**************************************************************
	// Kernel for calculating Moments of the distribution Functions
	// 	a. Zeroth moment: Density
	//	b. First moment: Velosity
	// for the Lattice Boltzmann algorithm

	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	//**************************************************************
	//==========================================================================================
	__global__ void GPU_CalcMacroVars(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_MacroVars, unsigned int nArr_dbl, long long lower_limit, long long upper_limit)
	{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind =Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

			//GMem_dbl_fOld_b[0] = fOld[19][nFluid_nodes]
			//GMem_dbl_fNew_b[0] = fNew[19][nFluid_nodes]

			//GMem[38*nArr] = density[nNodes]
			//GMem[39*nArr] = u[3][nNodes]

			//Read in the fNew[19][Ind] and copy back to fOld[19][Ind]
			double dev_ff[19];
			double Density = 0.0;
			double momentum_x, momentum_y, momentum_z;
			momentum_x = momentum_y = momentum_z = 0.0;

			double velx, vely, velz;	// Fluid Velocity

			// a. Read fOld_GPU_b
			// b. Calculates the density and momentum (velocity) at the begining of the LB loop, based on the old distribution functions ( time = t )

			for(int direction = 0; direction< _NUMVECTORS; direction++){
				dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction*nArr_dbl + Ind];	//fOld[i][Ind]
			}

			//Calculate density and momentum
			for(int direction = 0; direction< _NUMVECTORS; direction++){
				Density += dev_ff[direction];
				momentum_x += _CX_19[direction] * dev_ff[direction];
				momentum_y += _CY_19[direction] * dev_ff[direction];
				momentum_z += _CZ_19[direction] * dev_ff[direction];
			}

			//printf("Density = %.5e, Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", Density, momentum_x, momentum_y, momentum_z);

			// In the case of body force
			//momentum_x += 0.5 * _force_x;
			//momentum_y += 0.5 * _force_y;
			//momentum_z += 0.5 * _force_z;

			// Compute velocity components
			velx = momentum_x/Density;
			vely = momentum_y/Density;
			velz = momentum_z/Density;


			// Write updated density and velocity to memory
			GMem_dbl_MacroVars[Ind] = Density;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

		} // Ends the kernel GPUCalcMacroVars
		//==========================================================================================




	//==========================================================================================
	__global__ void GPU_CalcMacroVars_Swap(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_MacroVars, distribn_t* GMem_dbl_fNew_b, unsigned int nArr_dbl, long long lower_limit, long long upper_limit, int time_Step)
	{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind =Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

			//GMem_dbl_fOld_b[0] = fOld[19][nFluid_nodes]
			//GMem_dbl_fNew_b[0] = fNew[19][nFluid_nodes]

			//GMem[38*nArr] = density[nNodes]
			//GMem[39*nArr] = u[3][nNodes]

			//Read in the fNew[19][Ind] and copy back to fOld[19][Ind]
			double dev_ff[19];
			double Density = 0.0;
			double momentum_x, momentum_y, momentum_z;
			momentum_x = momentum_y = momentum_z = 0.0;

			double velx, vely, velz;	// Fluid Velocity

			// a. Calculates the density and momentum (velocity) based on the post-collision distribution functions (i.e. time = t+dt)
			// b. Swap the populations  - fNew in fOld
			for(int i=0; i< _NUMVECTORS; i++){
				// Read fNew
				dev_ff[i] = GMem_dbl_fNew_b[(unsigned long long)i*nArr_dbl + Ind];	//fNew[i][Ind]

				// Save fNew in fOld
				GMem_dbl_fOld_b[(unsigned long long)i*nArr_dbl + Ind] = dev_ff[i];	//fOld[i][Ind] = fNew[i][Ind]
			}

			//Calculate density and momentum
			for(int direction = 0; direction< _NUMVECTORS; direction++){
				Density += dev_ff[direction];
				momentum_x += _CX_19[direction] * dev_ff[direction];
				momentum_y += _CY_19[direction] * dev_ff[direction];
				momentum_z += _CZ_19[direction] * dev_ff[direction];
			}
			printf("Density = %.5e, Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", Density, momentum_x, momentum_y, momentum_z);

			// In the case of body force
			//momentum_x += 0.5 * _force_x;
			//momentum_y += 0.5 * _force_y;
			//momentum_z += 0.5 * _force_z;

			// Compute velocity components
			velx = momentum_x/Density;
			vely = momentum_y/Density;
			velz = momentum_z/Density;


			// Write updated density and velocity to memory
			GMem_dbl_MacroVars[Ind] = Density;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

		} // Ends the kernel GPUCalcMacroVars_Swap
		//==========================================================================================


		//==========================================================================================
		// Save the fNew post-collision distribution functions in the fOld array
		// Each thread is responsible for reading the fNew_GPU_b distr. functions for a lattice fluid node
		// i.e. the range for this kernel should be [0, nFluid_nodes) -
		// ***	 Does not swap the totalSharedFs distr. *** //
		// and then saves these values in fOld_GPU_b.
		// Check the discussion here:
		// https://stackoverflow.com/questions/22345391/cuda-device-memory-copies-cudamemcpydevicetodevice-vs-copy-kernel
		//==========================================================================================
		__global__ void GPU_SwapOldAndNew(distribn_t* __restrict__ GMem_dbl_fOld_b, distribn_t* __restrict__ GMem_dbl_fNew_b, site_t nArr_dbl, site_t lower_limit, site_t upper_limit)
		{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind =Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

			//GMem_dbl_fOld_b[0] = fOld[_NUMVECTORS][nFluid_nodes]
			//GMem_dbl_fNew_b[0] = fNew[_NUMVECTORS][nFluid_nodes]
			//printf("blockDim.x = %d, gridDim.x = %d, Product = %lld \n\n", blockDim.x, gridDim.x, blockDim.x * gridDim.x );

			for (int unsigned long long Index = Ind;
         Index < upper_limit;
         Index += blockDim.x * gridDim.x)
      	{
				// Just copy the populations  - fNew in fOld
				//Read in the fNew[19][Ind] and copy to fOld[19][Ind]
				for(int i=0; i< _NUMVECTORS; i++){
					GMem_dbl_fOld_b[(unsigned long long)i*nArr_dbl + Index] = GMem_dbl_fNew_b[(unsigned long long)i*nArr_dbl + Index];
				}
			}

		}	// Ends the GPU_SwapOldAndNew kernel
		//==========================================================================================


		//==========================================================================================
		// GPU kernel to do the appropriate re-allocation of the received distr. functions
		// placed in totalSharedFs in fOld in the RECEIVING rank (host-to-device memcpy preceded this kernel)
		// into the destination buffer "f_new"
		// using the streamingIndicesForReceivedDistributions (GPUDataAddr_int64_streamInd)
		// 		see: *GetFNew(streamingIndicesForReceivedDistributions[i]) = *GetFOld(neighbouringProcs[0].FirstSharedDistribution + i);
		// 		from LatticeData::CopyReceived()
		//==========================================================================================
		__global__ void GPU_StreamReceivedDistr(distribn_t* GMem_dbl_fOld_b, distribn_t* GMem_dbl_fNew_b, site_t* GMem_int64_streamInd, site_t nArr_dbl, site_t upper_limit)
		{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			// Ind =Ind + lower_limit; // limits are: for (site_t i = 0; i < totalSharedFs; i++)

			if(Ind >= upper_limit)
				return;

			//GMem_dbl_fOld_b = fOld[19][nFluid_nodes] then 1+totalSharedFs
			//GMem_dbl_fNew_b = fNew[19][nFluid_nodes] then 1+totalSharedFs

			//Read in the fOld[neighbouringProcs[0].FirstSharedDistribution + Ind] and then place this in the appropriate index in fNew
			distribn_t dev_fOld;
			dev_fOld = GMem_dbl_fOld_b[(unsigned long long)_NUMVECTORS * nArr_dbl + 1 + Ind];

			// Read the corresponding Index from the streaming Indices For Received Distributions
			// Remeber that this index refers to data layout method (a),
			//	i.e. Arrange by fluid index (as is hemeLB CPU version), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]
			// Need to convert to data layout method (b),
			// 	i.e. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., fq[0 to (nFluid_nodes-1)]
			site_t streamIndex_method_a = GMem_int64_streamInd[Ind];

			// Convert to data layout method (b)
			// 		The streamed array index (method_a) is within the domain, i.e. [0,nFluid_nodes*_NUMVECTORS)
			// 		a. The LB_dir, [0,_NUMVECTORS), will then be the value returned by modulo(_NUMVECTORS):
			int LB_Dir = streamIndex_method_a % _NUMVECTORS;
			// 		b. Fluid ID
			site_t fluid_ID = (streamIndex_method_a - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL fluid ID index

			site_t streamIndex_method_b = LB_Dir * nArr_dbl + fluid_ID;

			//printf("Index = %lld, LB_Dir = %d, Streamed Index_a = %lld, Streamed Index_b = %lld, fNew = %.5f \n\n", Ind, LB_Dir, streamIndex_method_a, streamIndex_method_b, dev_fOld);
			GMem_dbl_fNew_b[streamIndex_method_b] = dev_fOld;

		}	// Ends the GPU_StreamReceivedDistr kernel
		//==========================================================================================



		//==========================================================================================
		// 	Modification of the previous GPU kernel -Stream the received populations to fOld !!!
		// *****************************************************************************************
		//	GPU kernel to do the appropriate re-allocation of the received distr. functions
		// 		placed in totalSharedFs in fOld in the RECEIVING rank (host-to-device memcpy preceded this kernel)
		// 		into the destination buffer "f_old" : CHANGED from fNew
		// 			so that no swap of the distr. functions is needed.
		// 		using the streamingIndicesForReceivedDistributions (GPUDataAddr_int64_streamInd)
		// 		See: *GetFNew(streamingIndicesForReceivedDistributions[i]) = *GetFOld(neighbouringProcs[0].FirstSharedDistribution + i);
		// 					from LatticeData::CopyReceived()
		//	Note: Need to make sure that the collision-streaming kernels have completed their access to fOLd
		//==========================================================================================
		__global__ void GPU_StreamReceivedDistr_fOldTofOld(distribn_t* GMem_dbl_fOld_b, site_t* GMem_int64_streamInd, site_t nArr_dbl, site_t upper_limit)
		{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			// Ind =Ind + lower_limit; // limits are: for (site_t i = 0; i < totalSharedFs; i++)

			if(Ind >= upper_limit)
				return;

			//GMem_dbl_fOld_b = fOld[19][nFluid_nodes] then 1+totalSharedFs
			//GMem_dbl_fNew_b = fNew[19][nFluid_nodes] then 1+totalSharedFs

			//Read in the fOld[neighbouringProcs[0].FirstSharedDistribution + Ind] and then place this in the appropriate index in fNew
			distribn_t dev_fOld;
			dev_fOld = GMem_dbl_fOld_b[(unsigned long long)_NUMVECTORS * nArr_dbl + 1 + Ind];

			// Read the corresponding Index from the streaming Indices For Received Distributions
			// Remeber that this index refers to data layout method (a),
			//	i.e. Arrange by fluid index (as is hemeLB CPU version), i.e f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]
			// Need to convert to data layout method (b),
			// 	i.e. Arrange by index_LB, i.e. f0[0 to (nFluid_nodes-1)], f1[0 to (nFluid_nodes-1)], ..., fq[0 to (nFluid_nodes-1)]
			site_t streamIndex_method_a = GMem_int64_streamInd[Ind];

			// Convert to data layout method (b)
			// 		The streamed array index (method_a) is within the domain, i.e. [0,nFluid_nodes*_NUMVECTORS)
			// 		a. The LB_dir, [0,_NUMVECTORS), will then be the value returned by modulo(_NUMVECTORS):
			int LB_Dir = streamIndex_method_a % _NUMVECTORS;
			// 		b. Fluid ID
			site_t fluid_ID = (streamIndex_method_a - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL fluid ID index

			site_t streamIndex_method_b = LB_Dir * nArr_dbl + fluid_ID;

			//printf("Index = %lld, LB_Dir = %d, Streamed Index_a = %lld, Streamed Index_b = %lld, fNew = %.5f \n\n", Ind, LB_Dir, streamIndex_method_a, streamIndex_method_b, dev_fOld);
			GMem_dbl_fOld_b[streamIndex_method_b] = dev_fOld;

		}	// Ends the GPU_StreamReceivedDistr kernel
		//==========================================================================================

#endif
}
