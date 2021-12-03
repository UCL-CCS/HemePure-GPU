// This file is part of the GPU development for HemeLB
/*
//------------------------------------------------------------------------------
	HemeLB-GPU version 1.28.a
//----------------------------------------------------------------------------
*/


#include <stdio.h>

#ifdef HEMELB_USE_GPU
#include "cuda_kernels_def_decl/cuda_params.h"
#endif


namespace hemelb
{

#ifdef HEMELB_USE_GPU

	// GPU constant memory
	 __constant__ site_t _Iolets_Inlet_Edge[local_iolets_MaxSIZE];
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

	__device__ __constant__ double _EQMWEIGHTS_19[19];

	__constant__ int _CX_19[19];
	__constant__ int _CY_19[19];
	__constant__ int _CZ_19[19];

	__constant__ int _WriteStep = 1000;
	__constant__ int _Send_MacroVars_DtH = 1000; // Writing MacroVariables to GPU global memory (Sending MacroVariables calculated during the collision-streaming kernels to the GPU Global mem).


	//===================================================================================================================

	/**
	__global__ GPU kernels
	*/



	//**************************************************************
	/** Kernel for assessing the stability of the code
			Remember that the enum Stability is defined in SimulationState.h:
							enum Stability
							{
								UndefinedStability = -1,
								Unstable = 0,
								Stable = 1,
								StableAndConverged = 2
							};
			Initial value set to UndefinedStability(i.e. -1).

			*** CRITERION ***
			The kernel assesses the stability by:
			1. Examining whether f_new > 0.0
						SAME approach as the CPU version of hemeLB
			2. Consider in the future checking for NaNs values (maybe just the density will suffice)

			If unstable (see criterion above):
				flag d_Stability_flag set to 0 (global memory int*).
	*/
	//**************************************************************
	__global__ void GPU_Check_Stability(distribn_t* GMem_dbl_fOld_b,
																		distribn_t* GMem_dbl_fNew_b,
																		int* d_Stability_flag,
																		site_t nArr_dbl,
																		site_t lower_limit, site_t upper_limit,
																		int time_Step)
	{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind = Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

			int Stability_GPU = *d_Stability_flag;
			//printf("Site ID = %lld - Stability flag: %d \n\n", Ind, Stability_GPU);

			/** At first, follow the same approach as in the CPU version of hemeLB,
					i.e. examine whether the distribution functions are positive, see lb/StabilityTester.h
					//--------------------------------------------------------------------
					Also, see SimulationState.h for the enum Stability:
					namespace lb
  				{
    				enum Stability
    				{
				      UndefinedStability = -1,
				      Unstable = 0,
				      Stable = 1,
				      StableAndConverged = 2
				    };
				  }
					//--------------------------------------------------------------------
			// Note that by testing for value > 0.0, we also catch stray NaNs.
			if (! (value > 0.0))
			{
				mUpwardsStability = Unstable;
				break;
			}
			*/

			// Load the distribution functions fNew_GPU_b[19]
			// distribn_t dev_ff_new[19];

			for(int direction = 0; direction< _NUMVECTORS; direction++){
				distribn_t ff = GMem_dbl_fNew_b[(unsigned long long)direction * nArr_dbl + Ind];
				//dev_ff_new[direction] = ff;
				if (!(ff > 0.0)) // Unstable simulation
				{
					Stability_GPU = 0;
					*d_Stability_flag = 0;
					return;
				}
				if(Stability_GPU==0)
					return;


			} // Ends the loop over the LB-directions

			// Debugging test
			//if(time_Step%200 ==0) *d_Stability_flag = 0;

	} // Ends the kernel GPU_Check_Stability
	//==========================================================================================


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
		double dev_ff[19], dev_fEq[19];
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
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements

		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau;
		}


		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices

		// b. If within the limits for the mWallCollision
		//		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes
		//		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)

		site_t index_wall = nArr_dbl * _NUMVECTORS; // typedef int64_t site_t;

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Neighbouring index refers to the index to be streamed to in the global memory. Here it Refers to Data Address NOT THE STREAMING FLUID ID!!!

				// Is there a performance gain in choosing Option 1 over Option 2 or Option 3 below???
				// Option 1:
				if(dev_NeighInd == index_wall) // Wall Link
				{
					// Simple Bounce Back case:
					GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir
				}
				else{
					GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir]; 	// If neigh_d is selected
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
