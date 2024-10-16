// This file is part of the GPU development for HemeLB
/*
//------------------------------------------------------------------------------
	HemeLB-GPU version 2.3
//------------------------------------------------------------------------------
*/

/**
July 2022

Optimising the case of Vel BCs:
1. Instead of 3 components for the wall momentum
 		pass the single term correction term to the GPU global Memory
		Done!

		TODO: Optimise this further - decrease the memory footprint further.

		Done:
		1.1. Directly passing the correction term without saving to the propertyCache
					and then reading from there.
		1.2. Make the mcpy asynchronous (of the correction term to the GPU global memory - use the cuda stream of the appropriate GPU collision-streaming kernel)
		1.3. Pass the appropriate cuda stream as an argument to the memcpy function
					function:
					bool memCpy_HtD_GPUmem_WallMom_correction_cudaStream(site_t firstIndex, site_t siteCount, std::vector<double>& wallMom_Iolet, void *GPUDataAddr_wallMom, cudaStream_t ptrStream);
		1.4. Check that all instances are correctly set (Inlet / Outlet). Done!!!

		TODO:
		1.4. cHECK THE GHOST Density WAY OF ALLOCATING AND MemCopyING

//-----------------------------------
HemeLB-GPU version 2.1.

2. This is a copy of the HemeLB_GPU Cuda code with the optimisations applied after the GPU 2022 hackathon event.
   Path: /home/ioannis/Documents/UCL_project/GPU_hackathon_UK_2022/hemeLB-GPU/HemePure-GPU/src

   The above extra optimisations for the Vel. BCs case (point 1) did not make a major difference (tested on my laptop
   with the
   pipe.gmy geometry
   path: ~/Documents/UCL_project/code_development/code_Validation/cases/pipe_VelInFile_PresOut/GPU_version_testing

   irrespective of the location of the evaluation & asynch. memcpy of the correction term
   	(either in PreSend or in PreReceive).

   TODO: move all the relevant info on the GPU and evaluate correction term for the 3 different cases on the GPU...
   (LaddIolet Vel. BCs)


//-----------------------------------
HemeLB-GPU version 2.2.

Done: Improve the Vel. BCs cases
			1. move all the relevant info on the GPU and evaluate correction term for the 3 different cases on the GPU...
				(LaddIolet Vel. BCs)

				Done!!!:
				1.1. A. Coordinates of fluid sites at each iolet Section
								Function memCpy_HtD_GPUmem_Coords_Iolets
									performs the memcpy HtD for the coordinates of the iolets points
										(GPUDataAddr_Coords_Inlet_Edge;
										GPUDataAddr_Coords_InletWall_Edge;
										GPUDataAddr_Coords_Inlet_Inner;
										GPUDataAddr_Coords_InletWall_Inner;
										GPUDataAddr_Coords_Outlet_Edge;
										GPUDataAddr_Coords_OutletWall_Edge;
										GPUDataAddr_Coords_Outlet_Inner;
										GPUDataAddr_Coords_OutletWall_Inner;)
								Format (int64_t), x,y,z coordinates in consecutive manner
									for the fluid sites in the range Fluid index:[firstIndex, (firstIndex + siteCount)]

				1.2. B. velocityTable - Only useful for the Case: b. File
									mLatDat->GPUDataAddr_Inlet_velocityTable

				1.3. C. weights_table - only used for the case of Vel. BCs - File (subtype)
									Coordinates and weights ()

			 	1.4. 	Store the position of the iolets (x,y,z ) = (position_iolet.x, position_iolet.y, position_iolet.z) - type double


			2. Done!!!
			 		Fixed
					Bug in the details for iolets:
					The following may not contain the right info
						struct Iolets Inlet_Edge, Inlet_Inner, InletWall_Edge, InletWall_Inner;
						struct Iolets Outlet_Edge, Outlet_Inner, OutletWall_Edge, OutletWall_Inner;

						TODO: Check this info as well as the
									info in the corresponding other parameters, e.g.
										std::vector<site_t> Iolets_Inlet_Edge; 			// vector with Inlet IDs and range associated with PreSend collision-streaming Type 3 (mInletCollision)
										int n_LocalInlets_mInlet_Edge; 							// number of local Inlets involved during the PreSend mInletCollision collision - needed for the range of fluid sites involved
										int n_unique_LocalInlets_mInlet_Edge;				// number of unique local Inlets

						Seems that the above struct Iolets were not initialised, so when accessing info that should be just 0
											it returns non-zero values.
						Hence, more reliable (until I check the bug) to work with
											std::vector<site_t> Iolets_Inlet_Edge; 			// vector with Inlet IDs and range associated with PreSend collision-streaming Type 3 (mInletCollision)
											int n_LocalInlets_mInlet_Edge; 							// number of local Inlets involved during the PreSend mInletCollision collision - needed for the range of fluid sites involved
											int n_unique_LocalInlets_mInlet_Edge;
						Check after calling function identify_Range_iolets_ID in initialise_GPU().

						Comment: The bug (case of zero number of iolets due to non initialisation)
						 					does not propagate because we ensure that the appropriate GPU kernels are Called
											only if the number of fluid sites is not zero.
											However, if we rely on the struct Iolets in a general way there can be errors.
						ACTION:		FIX (but not a priority)

29 Dec 2022
TODO: In initialisation generate a map container with the fluid ID and the index in the weights_table to get the
			appropriate weight.
			Generate 4 such maps (Inlet_Edge, InletWall_Edge, Inlet_Inner, InletWall_Inner, etc
			and pass the corresponding array to the GPU
			with the array shifted index by the starting Fluid ID for that corresponding type of collision-streaming
			i.e. arrays:
				index_weightTable_Inlet_Edge
				index_weightTable_InletWall_Edge
				index_weightTable_Inlet_Inner
				index_weightTable_InletWall_Inner

			Done!!! Created vectors with this info and passed that to the GPU
							No need to search for the xyz coords (halfWay) for the appropriate weight

//-----------------------------------
Jan 2023
HemeLB-GPU version 2.2.b

		GPU kernel:	GPU_WallMom_correction_File_Weights_NoSearch
		Just reads that index and calculates the correction term on the GPU

//-----------------------------------
March 2023
	HemeLB-GPU version 2.2.c
		1. Optimised Vel BCs case - subtype File
			Compute the velocity momentum correction terms on the GPU - using a prefactor
			correction term pre-evaluated at initialisation
			Just need to mutiply with the max Veocity (info obtained from the velocity table)


	More details - Other tasks
	1. Continue with the tasks from v2_2b (reduce the memory requirements in initialisation associated
			with the Vel BCs - case File) and what is listed below:

			Feb 2023
			TODO:
		1. Vel BCs case:
			1.1. Decrease the size of velocityTable - timeSteps (no need to repeat values in the table if there is periodicity in the values)
						No need to have values for each timeStep saved
							int total_TimeSteps = mState->GetTotalTimeSteps();

						Note IMPORTANT !!!
								The number of values in velocityTable and the file that we read from CAN BE different
								hemeLB will interpolate to evaluate the velocity value (max Vel) based on actual timeStep and the time in the velocity file

			1.2. Decrease the size of velocityTable - n_Inlets (this is the number of total inlets and NOT the actual number of local rank inlets)
						Data_dbl_Inlet_velocityTable = new distribn_t[n_Inlets * (total_TimeSteps+1)];

						How to obtain the number of local iolets:
									That would be the sum of the following (NO!!! - This is Wrong as there can be overlap of the same iolets appearing in the 4 different types below)
											n_unique_LocalInlets_mInlet_Edge, n_unique_LocalInlets_mInletWall_Edge, n_unique_LocalInlets_mInlet, n_unique_LocalInlets_mInletWall
									or maybe just run the function identify_Range_iolets_ID with the range of fluid sites on the local rank

			1.3. Read the files needed (velocityTable and weights) only on the processors with iolets
						Status: Done!!

		2. Clean-up

		3. Wall shear stress magnitude evaluated on the GPU
				Need to send the following to the GPU memory:
				3.1.


		4. Copy Balint's changes in ReadGeometry for
		 			Faster I/O for reading Geometry Reader using MPI I/O
				Status: Done!!


		5. Investigate whether there is a bug with the Vel BCs case.
		 		Single Inlet - Outlet case works fine. Absolute match with the CPU version.

				Multi-inlet case - Slight deviation between CPU and GPU results.
				 	Check the following:
					5.1. Loop over the fluid sites and report/compare
								5.1.1. Vel. Weight (Compare the one on the GPU with the weight on the host)
								5.1.2. Max Velocity as obtained from the velocity Table. (compare host and GPU values)

								5.1.3. Check the kernel
								 				GPU_WallMom_correction_File_Weights_NoSearch

												At the end of
													apply_Vel_BCs_File_GetWallMom_correction

												that calculates the correction term
													distribn_t correction = 2. * LatticeType::EQMWEIGHTS[ii]
																	* (wallMom.x * LatticeType::CX[ii]
																			+ wallMom.y * LatticeType::CY[ii]
																			+ wallMom.z * LatticeType::CZ[ii]) / Cs2;

																			with wallMom being calculated in
																					LatticeVelocity InOutLetFileVelocity::GetVelocity(const LatticePosition& x,
																																						const LatticeTimeStep t) const
																					which is returning (if xyz point found)
																						v_tot = normal * weights_table.at(xyz) * velocityTable[t];
													get these values
													GMem_dbl_WallMom
													or
													GPUDataAddr_wallMom_correction_Inlet_Edge etc


													-------------------
													Version 2_1 works fine (Wall mom correction  calculated on the host)
													e.g.
													// Debugging - 24 April 2023
													// Directly getting the correction term without passing through propertyCache

													GetWallMom_correction_Direct(mInletCollision, start_Index_Inlet_Edge, site_Count_Inlet_Edge, propertyCache, wallMom_correction_Inlet_Edge_Direct);
													wallMom_correction_Inlet_Edge_Direct.resize(site_Count_Inlet_Edge*LatticeType::NUMVECTORS);

													// Function to allocate memory on the GPU's global memory for the wallMom
													//memCpy_HtD_GPUmem_WallMom_correction(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_correction_Inlet_Edge_Direct, GPUDataAddr_wallMom_correction_Inlet_Edge);
													memCpy_HtD_GPUmem_WallMom_correction_cudaStream(start_Index_Inlet_Edge, site_Count_Inlet_Edge, wallMom_correction_Inlet_Edge_Direct,
																																					GPUDataAddr_wallMom_correction_Inlet_Edge, Collide_Stream_PreSend_3);

													******************************
													Possible solution
													Create a function to return at Initialise_GPU a prefactor for the correction term without the
													velocity dependency
														  GetWallMom_prefactor_correction_Direct

													This works fine!!!

													2 possible functions:
													a. apply_Vel_BCs_File_GetWallMom_correction_ApprPref()
															Does the search for the Iolet ID using the function _determine_Iolet_ID

													b. apply_Vel_BCs_File_GetWallMom_correction_ApprPref_NoIoletIDsearch()
															Does not search for the iolet ID - passed as an argument to the GPU kernel (loop through the local iolets and their fluid sites range - see struct Iolets Iolets_Inlet_Edge etc)
																GPU_WallMom_correction_File_prefactor_NoIoletIDSearch
													******************************

													Note:
													 Allocation for the iolets details (ID and fluid sites range) has been removed from all the GPU Kernels
													 	This was causing problems with the porting to oneAPI process!!!

//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
May 2023
HemeLB-GPU version 2.2.d
	Evaluate wall shear stress on the GPU
//------------------------------------------------------------------------------
//------------------------------------------------------------------------------
Dec 2023
HemeLB-GPU version 2.3
   Implementing the checkpointing functionality
	i.e. add the option for restarting the simulations

	Fixed a bug for:
	a. reading the restart time from the input file.
	b. passing that restart time to the simulation so that the correct time-dependent value (e.g. max velocity) is applied to iolets

	There are now 2 options when restarting the simulation:
	1. If the restart time is not specified, the simulation will read from the XTR checkpointing file
			the last time the distribution functions were saved.
	2. The user can specify the restart time
		2.1. Then a search will be performed in the XTR checkpointing file (if the specific time or the next available greater timestep is available)
//------------------------------------------------------------------------------

//------------------------------------------------------------------------------
General things:
TODO:
1. Pass a boolean variable to the GPU collision-streaming kernels for the compressibility of the model
 			Note: MUltiply the correction term with the density if the LB model is Compressible
 			i.e. Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()

2. Case of Vel. BCs
		If there is periodicity in the values (reading the Velocity from Table),
		save memory instead of allocating as
			Data_dbl_Inlet_velocityTable = new distribn_t[n_Inlets * (total_TimeSteps+1)];
			see Initialise_GPU() in lb/lb.hpp
//------------------------------------------------------------------------------
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

	__constant__ bool _useWeightsFromFile;
	__constant__ double _iStressParameter;

	__constant__ int _InvDirections_19[19];

	__device__ __constant__ double _EQMWEIGHTS_19[19];

	__constant__ int _CX_19[19];
	__constant__ int _CY_19[19];
	__constant__ int _CZ_19[19];

	__constant__ int _WriteStep = 100;
	__constant__ int _Send_MacroVars_DtH = 100; // Writing MacroVariables to GPU global memory (Sending MacroVariables calculated during the collision-streaming kernels to the GPU Global mem).

	__constant__ double dev_smag_cnst;

	//===================================================================================================================

	/**
	__global__ GPU kernels
	*/

	//============================================================================
	// Apply the optimisations from the GPU hackathon 2022 on all the kernels in use
	// Optimisations:
	// 	I. 		Merge the loops (test first having 3 loops, before merging to have 2 loops in total)
	//	II.		Loop unrolling: #pragma unroll 19
	//	III.  Compile with the flag maxxregcount set to 100 (leads to no memory spills to GPU global memory)

	// Currently using the following GPU kernels:
	// 1. Stability check: GPU_Check_Stability
	// 2. Collision-streaming kernels:
	//		2.1. 	GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB						Done!!!
	//		2.2.	GPU_CollideStream_Iolets_NashZerothOrderPressure_v2								Done!!!
	//		2.3. 	GPU_CollideStream_Iolets_NashZerothOrderPressure									Done!!!
	//		2.4. 	GPU_CollideStream_wall_sBB_iolet_Nash															Done!!!
	//		2.5. 	GPU_CollideStream_wall_sBB_iolet_Nash_v2													Done!!!
	//		2.6. 	GPU_CollideStream_Iolets_Ladd_VelBCs															Done!!!
	//		2.7. 	GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs
	// 3. Case of Velocity BCs and subtype: File:
	//		3.1.	GPU_WallMom_correction_File_Weights_NoSearch											Done!!!
	//============================================================================


//------------------------------------------------------------------------------
//**************************************************************
/** GPU kernel for evaluating the wall momentum correction terms on the GPU
		For the Velocity BCs - LADDIOLET BCs

		Implement what is in the function
				LatticeVelocity InOutLetFileVelocity::GetVelocity
		Note that the arguments are: (halfWay, bValues->GetTimeStep())
				LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

			I. Case 1: No weights needed (useWeightsFromFile=false)
					TODO in a seperate GPU kernel

			II. Case 2: weights needed (useWeightsFromFile=true)
					Need to have the following information on the GPU:
					1. normal vector to the iolet
					2. LatticePosition& x : coordinates of the point
							LatticePosition: Vector3D<double>
							see function memCpy_HtD_GPUmem_Coords_Iolets which fills the following:
								type int64_t [x_coord, y_coord, z_coord, ...]
								void *GPUDataAddr_Coords_Inlet_Edge;
								void *GPUDataAddr_Coords_InletWall_Edge;
								void *GPUDataAddr_Coords_Inlet_Inner;
								void *GPUDataAddr_Coords_InletWall_Inner;
								void *GPUDataAddr_Coords_Outlet_Edge;
								void *GPUDataAddr_Coords_OutletWall_Edge;
								void *GPUDataAddr_Coords_Outlet_Inner;
								void *GPUDataAddr_Coords_OutletWall_Inner;

					3. LatticeTimeStep t : The time-step (type unsigned long)
								units.h:  typedef unsigned long LatticeTimeStep

					4. weights table
								std::map<std::vector<int>, double> weights_table;

							The CPU version gets the weight (mapped value) using the key value.
							Here, we have 2 options:
							4.1. Save the index of the key value (perform the search at Initialise_GPU)
											and then save these indices so that I can easily obtain the corresponding weight (mapped value)
											without performing the search at each time-step
										switch to version 1 for this
											OR
							4.2. Perform the search on the GPU (at each time) of the key value (xyz coords)
											in order to obtain the corresponding weight (mapped value)
										switch to version 0

					5. velocityTable[t]

					6. arr_elementsInEachInlet[index_inlet] - n_arr_elementsInCurrentInlet_weightsTable
								number of elements (lines) in the weights table
								Note there is 1 such table for each iolet

					7. Consider moving this info in shared memory
									GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2];
									GMem_pp_dbl_weightsTable_wei[inlet_ID][ii];

			//---------------------
			Evaluate a single value correction term for each Fluid site and
				each of the directions (LB directions from 1 to 18, excluding the 0th LB dir)
			i.e. what is below the value of correction:

				LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

				distribn_t correction = 2. * LatticeType::EQMWEIGHTS[ii]
						* (wallMom.x * LatticeType::CX[ii]
								+ wallMom.y * LatticeType::CY[ii]
								+ wallMom.z * LatticeType::CZ[ii]) / Cs2;
			//---------------------

			Consider that the loop is going through the lattice points
			hence, we need to be able to determine the iolet ID from the fluid index

			TODO: pass the arr_elementsInEachInlet[index_inlet] to the GPU global memory
*/
//**************************************************************
//template<int version>
__global__ void GPU_WallMom_correction_File_Weights(int64_t *GMem_Coords_iolets,
																	int64_t **GMem_pp_int_weightsTable_coord,
																	distribn_t **GMem_pp_dbl_weightsTable_wei,
																	int64_t* GMem_index_key_weightTable,
																	distribn_t *GMem_dbl_WallMom,
																	float* GMem_ioletNormal,
																	uint32_t* GMem_uint32_Iolet_Link,
																	int inlet_ID,
																	distribn_t* GMem_Inlet_velocityTable,
																	int n_arr_elementsInCurrentInlet_weightsTable,
																	site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
																	site_t lower_limit, site_t upper_limit,
																	unsigned long time_Step, unsigned long total_TimeSteps)
{
	//extern __shared__ int64_t buf_coords[]; // Switching to this when using shared memory

	unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;

	Ind = Ind + lower_limit;

	if(Ind >= upper_limit)
		return;

	/*
	// This will be included into version 0 (using shared memory to perform search for the map key value)
	int shifted_Ind = Consider how to arrange this so that in each block all the elements in weights_table can be read in Shared memory
	buf_coords[shifted_Ind] = GMem_pp_int_weightsTable_coord[inlet_ID][shifted_Ind];

	__syncthreads(); // The __syncthreads() command is a block level synchronization barrier.
	// Hence, is there a way to ensure that all the elements in the block have read all the elements in the coords???
	// Have each thread of the total of nThreadsPerBlock_WallMom_correct (possibility that n_count =  (upper_limit - lower_limit +1) is lower than nThreadsPerBlock_WallMom_correct)
	// 	to read (n_arr_elementsInCurrentInlet_weightsTable / nThreadsPerBlock_WallMom_correct)

	// Load the coordinates in GMem_pp_int_weightsTable_coord
	// that will be searched, in shared memory
	// 	GMem_pp_int_weightsTable_coord[inlet_ID][]
	*/

		// TODO:
		/* Load the following info:
		1. Coordinates of the local point
				Done!!!
		2. normal vector to the iolet
				in the arguments above add the following:
				float* GMem_ioletNormal,
				Done!!!

		3. Check why in the CPU code the normal is declared as double - not float

		4. Load the iolet link info (which direction has a link to an iolet)
				Done!!!

				4.1. Check whether to keep the loop from LB_Dir=0 or 1
		*/

		// 1. Load the coordinates of the point for which we would like to evaluate the correction terms
		// Have in mind that (save registers per thread):
		// int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
		int64_t x_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3];
		int64_t y_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3 + 1];
		int64_t z_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3 + 2];
		//printf("Fluid Index = %lld,  Shifted Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, shifted_Fluid_Ind, x_coord, y_coord, z_coord);
		//printf("Fluid Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, x_coord, y_coord, z_coord);

		// Note that the normal vector components are of type float: float* GMem_ioletNormal
		double inletNormal_x = (double)GMem_ioletNormal[3*inlet_ID];
		double inletNormal_y = (double)GMem_ioletNormal[3*inlet_ID+1];
		double inletNormal_z = (double)GMem_ioletNormal[3*inlet_ID+2];
		//printf("Inlet ID: %d, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", inlet_ID, inletNormal_x, inletNormal_y, inletNormal_z);


		//==========================================================================
		// Load the Iolet-Fluid links info
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// Here is the loop over the LB lattice directions
#pragma unroll 19
			for (int LB_Dir = 1; LB_Dir < _NUMVECTORS; LB_Dir++) // Check whether to keep the loop from LB_Dir=0 or 1
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				distribn_t correction = 0.0;
				distribn_t vel_weight=0.0;
				distribn_t max_vel=0.0;
				distribn_t wallMom_x=0.0; distribn_t wallMom_y=0.0; distribn_t wallMom_z=0.0;

				// Check which links are fluid-iolet links
				if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
					//printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);
					//------------------------------------------------------------------
					// A. Step: ioletLinkDelegate.Eval_wallMom_correction(lbmParams, latDat, site, hydroVars, direction, &wallMom_correction_received);

						double halfWay_x = x_coord;
						double halfWay_y = y_coord;
						double halfWay_z = z_coord;

						halfWay_x += 0.5 * _CX_19[LB_Dir];
						halfWay_y += 0.5 * _CY_19[LB_Dir];
						halfWay_z += 0.5 * _CZ_19[LB_Dir];

						//------------------------------------------------------------------
						// B. Step: LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

						//==================================================================
						/** Continue here what is in GetVelocity
									Remember that the arguments are: GetVelocity(halfWay, bValues->GetTimeStep())
						 				LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
								TODO:
								Move what is independent of the LB_Dir direction out of the loop later
						*/
						double abs_normal[3] = {inletNormal_x, inletNormal_y, inletNormal_z};

						/* These absolute normal values can still be negative here,
						 * but are corrected below to become positive. */

						// prevent division by 0 errors if the normals are 0.0
						if (inletNormal_x < 0.0000001) { abs_normal[0] = 0.0000001; }
						if (inletNormal_y < 0.0000001) { abs_normal[1] = 0.0000001; }
						if (inletNormal_z < 0.0000001) { abs_normal[2] = 0.0000001; }

						int xyz_directions[3] = { 1, 1, 1 };
						int xyz[3] = { 0, 0, 0 };
						double xyz_residual[3] = {0.0, 0.0, 0.0};
						/* The residual values increase by the normal values at every time step. When they hit >1.0, then
			     	* xyz is incremented and a new grid point is attempted.
			     	* In addition, the specific residual value is decreased by 1.0. */

						if (inletNormal_x < 0.0)
					  {
					    xyz_directions[0] = -1;
					    // Fix this line: xyz[0] = floor(halfWay_x);
							xyz[0] = floor(halfWay_x);
					    abs_normal[0] = -abs_normal[0];
					    // start with a negative residual because we already moved partially in this direction
					    xyz_residual[0] = -(halfWay_x - floor(halfWay_x));
					  } else {
					    // Fix this line: xyz[0] = std::ceil(halfWay_x);
							xyz[0] = ceil(halfWay_x);
							xyz_residual[0] = -(ceil(halfWay_x) - halfWay_x);
					  }

						if (inletNormal_y < 0.0)
				    {
				      xyz_directions[1] = -1;
				      xyz[1] = floor(halfWay_y);
				      abs_normal[1] = -abs_normal[1];
				      xyz_residual[1] = -(halfWay_y - floor(halfWay_y));
				    } else {
				      xyz[1] = ceil(halfWay_y);
				      xyz_residual[1] = -(ceil(halfWay_y) - halfWay_y);
				    }

						if (inletNormal_z < 0.0)
				    {
				      xyz_directions[2] = -1;
				      xyz[2] = floor(halfWay_z);
				      abs_normal[2] = -abs_normal[2];
				      xyz_residual[2] = -(halfWay_z - floor(halfWay_z));
				    } else {
				      xyz[2] = ceil(halfWay_z);
				      xyz_residual[2] = -(ceil(halfWay_z) - halfWay_z);
				    }
						//printf("Fluid Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld), LB_Dir = %d, xyz = (%d, %d, %d) - Residuals : (%f, %f, %f) \n", Ind, x_coord, y_coord, z_coord, LB_Dir, xyz[0], xyz[1], xyz[2], xyz_residual[0], xyz_residual[1], xyz_residual[2]);

						//LatticeVelocity v_tot = 0; - This is the wall momentum, hence just use that term only (wallMom_x etc)
						double v_tot_x=0.0; double v_tot_y=0.0; double v_tot_z=0.0;
						int iterations = 0;

						//------------------------------------------------------------------
						// Approach 1:
						// 		Search for the key xyz in the file
						//			GMem_pp_int_weightsTable_coord[inlet_ID]
						//		to obtain the index index_key_weights_table. NOTE that data are ORDERED

						bool found_element_key=0;
						while (iterations < 3)
    				{
      				// 1. Search with the existing xyz key - if found set found_element_key=1 and call return
							// Call the search function here to obtain the index index_key_weights_table
							if (found_element_key==1) return;
							// 2. otherwise continue the while loop

							// printf("Fluid Index = %lld, iter: %d, LB_Dir = %d, Searching for key: xyz = (%d, %d, %d) \n", Ind, iterations, LB_Dir, xyz[0], xyz[1], xyz[2]);
							/*if (weights_table.count(xyz) > 0)
      				{
        				v_tot = normal * weights_table.at(xyz) * velocityTable[t];
        				return v_tot;
      				}*/


							// If the key used does not return a result... continue
							// propagate residuals to the move to the next grid point
      				double xstep = (1.0 - xyz_residual[0]) / abs_normal[0];
				      double ystep = (1.0 - xyz_residual[1]) / abs_normal[1];
				      double zstep = (1.0 - xyz_residual[2]) / abs_normal[2];

							double all_step = 0.0;
							int xyz_change = 0;

				      if(xstep < ystep) {
				        if (xstep < zstep) {
				          all_step = xstep;
				          xyz_change = 0;
				        } else {
				          if (ystep < zstep) {
				            all_step = ystep;
				            xyz_change = 1;
				          } else {
				            all_step = zstep;
				            xyz_change = 2;
				          }
				        }
				      } else {
				        if (ystep < zstep) {
				          all_step = ystep;
				          xyz_change = 1;
				        } else {
				          all_step = zstep;
				          xyz_change = 2;
				        }
				      }

							xyz_residual[0] += abs_normal[0] * all_step;
				      xyz_residual[1] += abs_normal[1] * all_step;
				      xyz_residual[2] += abs_normal[2] * all_step;

				      xyz[xyz_change] += xyz_directions[xyz_change];

				      xyz_residual[xyz_change] -= 1.0;

							iterations++;
						}

						if (found_element_key==1){

						}else{

						}

						// Approach 2:
						// 	Obtain the index of the corresponding coordinates (xyz) in the weights_table
						// 	i.e. evaluate the index at initialisation, since it only depends on the
						//	 geometry and then read this at every time step to get the correction term
						int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
						int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

						int64_t index_key_weights_table = GMem_index_key_weightTable[index];
						//if(Ind==10025) printf("Ind: %lld, index_in_key_weightTable: %d, index_key_weights_table: %lld, INT_MAX: %lld \n", Ind, index, index_key_weights_table, INT64_MAX);

						if (index_key_weights_table==INT_MAX){
							//printf("Error while getting the key index from GPU Global memory ... \n");
							correction = 0.0;
							int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

							//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
							//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));
							int index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
							GMem_dbl_WallMom[index_wallMom_correction] = correction;
							return;
						}

						//------------------------------------------------------------------
						// Wall momentum returned:
						// 		v_tot = normal * weights_table.at(xyz) * velocityTable[t];
						vel_weight = GMem_pp_dbl_weightsTable_wei[inlet_ID][index_key_weights_table];

						max_vel = GMem_Inlet_velocityTable[inlet_ID *(total_TimeSteps+1) + time_Step]; // index_inlet*(total_TimeSteps+1)+timeStep


						if(time_Step>=4300 &&  time_Step <4310 && Ind==lower_limit) {
							printf("From GPU - velocityTable[timeStep = %d] = %0.8e, InletID: %d \n", time_Step, max_vel, inlet_ID);
						}

						wallMom_x = inletNormal_x * vel_weight * max_vel;
						wallMom_y = inletNormal_y * vel_weight * max_vel;
						wallMom_z = inletNormal_z * vel_weight * max_vel;

						//------------------------------------------------------------------
						// C. Step: Evaluate the single correction term as
						correction = 2. * _EQMWEIGHTS_19[LB_Dir]
				                * (	wallMom_x * (double)_CX_19[LB_Dir] +
														wallMom_y * (double)_CY_19[LB_Dir] +
				                    wallMom_z * (double)_CZ_19[LB_Dir]) / _Cs2;

						// if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

						// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
						// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
						// 	*** Do that inside the collision-streaming kernels Instead ***
						//		correction *= nn;
						//==================================================================

				} // Closes the loop if(is_Iolet_link)
				else{ // Don't really need this extra piece of code (else statement unnecessary)
					correction = 0.0;
				}

				//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %f \n", Ind, LB_Dir, correction);

				// Save the correction term in GPU global memory
				// TODO: Need to pass the:
				//		siteCount and shifted_Fluid_Ind
				// siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
				// shifted_Fluid_Ind is
				int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

				//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
				//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));
				int index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
				GMem_dbl_WallMom[index_wallMom_correction] = correction;

			} // ends the loop over the LB_Dir directions
			//==========================================================================

			//------------------------------------------------------------------------
			//------------------------------------------------------------------------
			/*// Debugging - Remove later - Data are ORDERED - confirmed
			if(Ind==10000){
				for (int ii=0; ii<n_arr_elementsInCurrentInlet_weightsTable; ii++)
				//for (int ii=0; ii<1; ii++)
				{
					// Data-layout xyz together
					//int64_t x_coord_weights = GMem_pp_int_weightsTable_coord[inlet_ID][ii*3];
					//int64_t y_coord_weights = GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+1];
					//int64_t z_coord_weights = GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2];

					// Data-layout xyz separately (x first, then y, finally z)
					int64_t x_coord_weights = GMem_pp_int_weightsTable_coord[inlet_ID][ii];
					int64_t y_coord_weights = GMem_pp_int_weightsTable_coord[inlet_ID][1*n_arr_elementsInCurrentInlet_weightsTable + ii];
					int64_t z_coord_weights = GMem_pp_int_weightsTable_coord[inlet_ID][2*n_arr_elementsInCurrentInlet_weightsTable + ii];
					distribn_t vel_weight = GMem_pp_dbl_weightsTable_wei[inlet_ID][ii];

					printf("GPU - Coordinates (x,y,z) : (%lld, %lld, %lld ) - Weight: %f \n", x_coord_weights,
																																											y_coord_weights,
																																											z_coord_weights,
																																											vel_weight);
						//printf("Velocity Table : %f \n", GMem_Inlet_velocityTable[ii]);
				}
		} // Ends the if (Ind==1000)
		*/
		//------------------------------------------------------------------------

}




//**************************************************************
/** GPU kernel for evaluating the wall momentum correction terms on the GPU
		1. 	For the Velocity BCs - LADDIOLET BCs
		1.1	Case 2: weights needed (useWeightsFromFile=true)
					Implement what is in the function
						LatticeVelocity InOutLetFileVelocity::GetVelocity
					Note that the arguments are: (halfWay, bValues->GetTimeStep())
						LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
		1.2. NO NEED TO SEARCH for the appropriate weight based on the coords (halfway) -

					Approach 1: Perform the search
					Approach 2: Read this information (which element to access)

				Remove all the unnecessary elements (from the kernel GPU_WallMom_correction_File_Weights above copied below now)...

					Need to have the following information on the GPU:
					1. normal vector to the iolet
					2. LatticePosition& x : coordinates of the point
							LatticePosition: Vector3D<double>
							see function memCpy_HtD_GPUmem_Coords_Iolets which fills the following:
								type int64_t [x_coord, y_coord, z_coord, ...]
								void *GPUDataAddr_Coords_Inlet_Edge;
								void *GPUDataAddr_Coords_InletWall_Edge;
								void *GPUDataAddr_Coords_Inlet_Inner;
								void *GPUDataAddr_Coords_InletWall_Inner;
								void *GPUDataAddr_Coords_Outlet_Edge;
								void *GPUDataAddr_Coords_OutletWall_Edge;
								void *GPUDataAddr_Coords_Outlet_Inner;
								void *GPUDataAddr_Coords_OutletWall_Inner;

					3. LatticeTimeStep t : The time-step (type unsigned long)
								units.h:  typedef unsigned long LatticeTimeStep

					4. weights table
								std::map<std::vector<int>, double> weights_table;

							The CPU version gets the weight (mapped value) using the key value.
							Here, we have 2 options:
							4.1. Save the index of the key value (perform the search at Initialise_GPU)
											and then save these indices so that I can easily obtain the corresponding weight (mapped value)
											without performing the search at each time-step
										switch to version 1 for this
											OR
							4.2. Perform the search on the GPU (at each time) of the key value (xyz coords)
											in order to obtain the corresponding weight (mapped value)
										switch to version 0

					5. velocityTable[t]

					6. arr_elementsInEachInlet[index_inlet] - n_arr_elementsInCurrentInlet_weightsTable
								number of elements (lines) in the weights table
								Note there is 1 such table for each iolet

					7. Consider moving this info in shared memory
									GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2];
									GMem_pp_dbl_weightsTable_wei[inlet_ID][ii];

			//---------------------
			Evaluate a single value correction term for each Fluid site and
				each of the directions (LB directions from 1 to 18, excluding the 0th LB dir)
			i.e. what is below the value of correction:

				LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));

				distribn_t correction = 2. * LatticeType::EQMWEIGHTS[ii]
						* (wallMom.x * LatticeType::CX[ii]
								+ wallMom.y * LatticeType::CY[ii]
								+ wallMom.z * LatticeType::CZ[ii]) / Cs2;
			//---------------------

			Consider that the loop is going through the lattice points
			hence, we need to be able to determine the iolet ID from the fluid index

			TODO: pass the arr_elementsInEachInlet[index_inlet] to the GPU global memory
*/
//**************************************************************
__global__ void GPU_WallMom_correction_File_Weights_NoSearch(int64_t *GMem_Coords_iolets,
		int64_t **GMem_pp_int_weightsTable_coord,
		distribn_t **GMem_pp_dbl_weightsTable_wei,
		int64_t* GMem_index_key_weightTable,
		distribn_t* GMem_weightTable,
		distribn_t *GMem_dbl_WallMom,
		float* GMem_ioletNormal,
		uint32_t* GMem_uint32_Iolet_Link,
		int inlet_ID,
		distribn_t* GMem_Inlet_velocityTable,
		int n_arr_elementsInCurrentInlet_weightsTable,
		site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
		site_t lower_limit, site_t upper_limit,
		unsigned long time_Step, unsigned long total_TimeSteps, unsigned long start_time)
{
	unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;

	Ind = Ind + lower_limit;

	if(Ind >= (upper_limit+1))
		return;

		// Done:
		/* Load the following info:
		1. Coordinates of the local point
				Done!!! - Not needed with Approach 2
		2. normal vector to the iolet
				in the arguments above add the following:
				float* GMem_ioletNormal,
		3. Check why in the CPU code the normal is declared as double - not float
		4. Load the iolet link info (which direction has a link to an iolet)
		*/

		// 1. Load the coordinates of the point for which we would like to evaluate the correction terms
		// Have in mind that (save registers per thread):
		// int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
		/* NOT NEEDED:
		int64_t x_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3];
		int64_t y_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3 + 1];
		int64_t z_coord = GMem_Coords_iolets[(Ind - start_Fluid_ID_givenColStreamType)*3 + 2];
		//printf("Fluid Index = %lld,  Shifted Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, shifted_Fluid_Ind, x_coord, y_coord, z_coord);
		//printf("Fluid Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, x_coord, y_coord, z_coord);
		*/

		// Note that the normal vector components are of type float: float* GMem_ioletNormal
		double inletNormal_x = (double)GMem_ioletNormal[3*inlet_ID];
		double inletNormal_y = (double)GMem_ioletNormal[3*inlet_ID+1];
		double inletNormal_z = (double)GMem_ioletNormal[3*inlet_ID+2];
		//printf("Inlet ID: %d, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", inlet_ID, inletNormal_x, inletNormal_y, inletNormal_z);

		//==========================================================================
		// Load the Iolet-Fluid links info
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// Here is the loop over the LB lattice directions
#pragma unroll 19
			for (int LB_Dir = 1; LB_Dir < _NUMVECTORS; LB_Dir++) // keep the loop from LB_Dir=1
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				distribn_t correction = 0.0;
				distribn_t vel_weight=0.0;
				distribn_t vel_weight_1=0.0; 	// Reading the index from the host (precalculated at Initialise_GPU)
				distribn_t vel_weight_2=0.0;	// Reading the actual vel weight (precalculated at Initialise_GPU)

				distribn_t max_vel=0.0;
				distribn_t wallMom_x=0.0; distribn_t wallMom_y=0.0; distribn_t wallMom_z=0.0;

				int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
				int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

				// Check which links are fluid-iolet links
				if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
					//printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

					// Step: LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
						/** Continue here what is in GetVelocity
									Remember that the arguments are: GetVelocity(halfWay, bValues->GetTimeStep())
						 				LatticeVelocity wallMom(iolet->GetVelocity(halfWay, bValues->GetTimeStep()));
									No need to Search for the halfWay xyz coordinates in the velocity weights table - Just read the index!
						*/

						/*
						// Approach 2:
						// 	Obtain the index of the corresponding coordinates (xyz) in the weights_table
						// 	i.e. evaluate the index at initialisation, since it only depends on the
						//	 geometry and then read this at every time step to get the correction term
						int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
						int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

						int64_t index_key_weights_table = GMem_index_key_weightTable[index];
						//if(Ind==10025) printf("Ind: %lld, index_in_key_weightTable: %d, index_key_weights_table: %lld, INT_MAX: %lld \n", Ind, index, index_key_weights_table, INT64_MAX);

						// Corresponds to the cases that do not find the xyz coord in the loop: while (iterations < 3) in LatticeVelocity InOutLetFileVelocity::GetVelocity
						// In which case the correction term is set to 0
						if (index_key_weights_table==INT_MAX){
							//printf("No xyz coord match in function LatticeVelocity InOutLetFileVelocity::GetVelocity ... \n");
							correction = 0.0;
							//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
							//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));

							int index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
							GMem_dbl_WallMom[index_wallMom_correction] = correction; // Maybe NOT needed - Check - TODO
							return;
						}

						if (index_key_weights_table==INT_MAX-1){
							printf("Shouldn't be in this loop ... Error with the iolet links and while getting the key index from GPU Global memory ... \n");
						}

						//------------------------------------------------------------------
						// Wall momentum returned:
						// 		v_tot = normal * weights_table.at(xyz) * velocityTable[t];

						// April 2023
						// Initial Approach that resulted in error for multi-inlet case
						vel_weight_1 = GMem_pp_dbl_weightsTable_wei[inlet_ID][index_key_weights_table];
						*/
						// New Approach - Get the Vel - weight directly
						vel_weight_2 = GMem_weightTable[index];

						//if (time_Step==1) // && vel_weight_1!= vel_weight_2)
						//	printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight(1): %5.3e, vel_weight(2): %5.3e \n", Ind, LB_Dir, vel_weight_1, vel_weight_2);

						vel_weight = vel_weight_2;
						//------------------------------------------------------------------

						max_vel = GMem_Inlet_velocityTable[inlet_ID *(total_TimeSteps+1) + time_Step - start_time]; // index_inlet*(total_TimeSteps+1)+timeStep

						wallMom_x = inletNormal_x * vel_weight * max_vel;
						wallMom_y = inletNormal_y * vel_weight * max_vel;
						wallMom_z = inletNormal_z * vel_weight * max_vel;

						//------------------------------------------------------------------
						// C. Step: Evaluate the single correction term as
						correction = 2. * _EQMWEIGHTS_19[LB_Dir]
				                * (	wallMom_x * (double)_CX_19[LB_Dir] +
														wallMom_y * (double)_CY_19[LB_Dir] +
				                    wallMom_z * (double)_CZ_19[LB_Dir]) / _Cs2;

						// if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

						// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
						// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
						// 	*** Do that inside the collision-streaming kernels Instead ***
						//		correction *= nn;
						//==================================================================

						//
						// Save the correction term in GPU global memory
						// TODO: Need to pass the:
						//		siteCount and shifted_Fluid_Ind
						// siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
						// shifted_Fluid_Ind is
						//int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

						//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
						//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));

						int index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
						GMem_dbl_WallMom[index_wallMom_correction] = correction;
						//
				} // Closes the loop if(is_Iolet_link)

				//if (time_Step==1 && correction!=0)
					//printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);


			} // ends the loop over the LB_Dir directions
			//==========================================================================

} // Ends the GPU kernel



//**************************************************************
/* April 2023
		New kernel for evaluating the wall momentum corection terms
			on the GPU.

		Approach:
			A. Use the geometric prefactor associated with the wall momentum
					correction terms.
					See ...
			B. Just multiply with the maximum velocity(t) for the Corresponding
						iolet from the velocity table

		Identify the iolet index from the fluid index and info on the GPU global memory
			site_t* GMem_Iolets_info containing:
		iolet ID and fluid sites range

		Consider moving the above information in GPU constant memory
*/
//**************************************************************
//**************************************************************
__global__ void GPU_WallMom_correction_File_prefactor(
											distribn_t* GMem_dbl_wallMom_prefactor_correction,
											distribn_t* GMem_dbl_WallMom,
											uint32_t* GMem_uint32_Iolet_Link,
											int num_local_Iolets, site_t* GMem_Iolets_info,
											distribn_t* GMem_Inlet_velocityTable,
											site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
											site_t lower_limit, site_t upper_limit,
											unsigned long time_Step, unsigned long total_TimeSteps)
{
	unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;

	Ind = Ind + lower_limit;

	if(Ind >= upper_limit)
		return;

		/** Load the following info:
					I. 		Velocity table Information - Requires iolet ID
					II. 	Identify the local iolet (to be used in the velocity table)
					III. 	Iolet link info (which direction has a link to an iolet)
					IV. 	WallMom prefactor correction term
		*/

		//--------------------------------------------------------------------------
		// II. 	Identify the local iolet
		//  		There are 2 possible ways:
		// 			1. Using the information from GPU global mem (GMem_Iolets_info)
		//			2. Using struct Iolets containing the info (when number of iolets less than 30)

		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			// Approach 1 - from GPU global mem (GMem_Iolets_info)
			IdInlet = GMem_Iolets_info[0];
			// Approach 2 - from struct array
			// IdInlet = Iolets_info.Iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			//	Contains the following (in the order below)
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)

			// Approach 1 - from GPU global mem (GMem_Iolets_info)
			_determine_Iolet_ID(num_local_Iolets, GMem_Iolets_info, Ind, &IdInlet);

			// Approach 2 - from struct array
			// _determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
		}

		// Debugging:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! FAILURE!!! Needs to abort... \n\n", Ind, IdInlet);
		}
		/*else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}*/
		//--------------------------------------------------------------------------

		//==========================================================================
		// III. Load the Iolet-Fluid links info
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// Here is the loop over the LB lattice directions
#pragma unroll 19
			for (int LB_Dir = 1; LB_Dir < _NUMVECTORS; LB_Dir++) // keep the loop from LB_Dir=1
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				distribn_t correction = 0.0;
				distribn_t max_vel=0.0; // Max Velocity to be read from the velocityTable(IdInlet,t)

				int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
				// int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

				// Check which links are fluid-iolet links
				if(is_Iolet_link){
					//printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

					// A. Step: Load the prefactor correction term
					//	1. TODO
					site_t index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
					distribn_t prefactor_correction = GMem_dbl_wallMom_prefactor_correction[index_wallMom_correction];

					// Note that the prefactor contains the info (see Eval_wallMom_prefactor_correction in LaddIoletDelegate)
					/**
							LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));

							distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
                			* (wallMom_prefactor.x * LatticeType::CX[ii] + wallMom_prefactor.y * LatticeType::CY[ii]
                    		+ wallMom_prefactor.z * LatticeType::CZ[ii]) / Cs2;
					*/

					// Just multiply with max Velocity(IdInlet,t) from velocityTable
					// 	Load max Vel
					max_vel = GMem_Inlet_velocityTable[IdInlet *(total_TimeSteps+1) + time_Step]; // index_inlet*(total_TimeSteps+1)+timeStep

					// B. Step: Evaluate the single correction term as
					correction = prefactor_correction * max_vel;


						/*wallMom_x = inletNormal_x * vel_weight * max_vel;
						wallMom_y = inletNormal_y * vel_weight * max_vel;
						wallMom_z = inletNormal_z * vel_weight * max_vel;

						//------------------------------------------------------------------
						// C. Step: Evaluate the single correction term as
						correction = 2. * _EQMWEIGHTS_19[LB_Dir]
				                * (	wallMom_x * (double)_CX_19[LB_Dir] +
														wallMom_y * (double)_CY_19[LB_Dir] +
				                    wallMom_z * (double)_CZ_19[LB_Dir]) / _Cs2;
						*/

						// if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

						// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
						// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
						// 	*** Do that inside the collision-streaming kernels Instead ***
						//		correction *= nn;
						//==================================================================

						//
						// Save the correction term in GPU global memory
						// TODO: Need to pass the:
						//		siteCount and shifted_Fluid_Ind
						// siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
						// shifted_Fluid_Ind is
						//int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

						//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
						//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));

						GMem_dbl_WallMom[index_wallMom_correction] = correction;
						//
				} // Closes the loop if(is_Iolet_link)

				//if (time_Step==1 && correction!=0)
					//printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);


			} // ends the loop over the LB_Dir directions
			//==========================================================================

} // Ends the GPU kernel

//**************************************************************
/* April 2023
		New kernel for evaluating the wall momentum corection terms
			on the GPU.

		Approach:
			A. Use the geometric prefactor associated with the wall momentum
					correction terms.
					See ...
			B. Just multiply with the maximum velocity(t) for the Corresponding
						iolet from the velocity table

		Identify the iolet index from the fluid index and info passed as an argument to the GPU kernel
			Iolets Iolets_info containing:
		iolet ID and fluid sites range

		Consider moving the above information in GPU constant memory
*/
//**************************************************************
//**************************************************************
__global__ void GPU_WallMom_correction_File_prefactor_v2(
											distribn_t* GMem_dbl_wallMom_prefactor_correction,
											distribn_t* GMem_dbl_WallMom,
											uint32_t* GMem_uint32_Iolet_Link,
											int num_local_Iolets, Iolets Iolets_info,
											distribn_t* GMem_Inlet_velocityTable,
											site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
											site_t lower_limit, site_t upper_limit,
											unsigned long time_Step, unsigned long total_TimeSteps)
{
	unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;

	Ind = Ind + lower_limit;

	if(Ind >= upper_limit)
		return;

		/** Load the following info:
					I. 		Velocity table Information - Requires iolet ID
					II. 	Identify the local iolet (to be used in the velocity table)
					III. 	Iolet link info (which direction has a link to an iolet)
					IV. 	WallMom prefactor correction term
		*/

		//--------------------------------------------------------------------------
		// II. 	Identify the local iolet
		//  		There are 2 possible ways:
		// 			1. Using the information from GPU global mem.
		//			2. Using struct Iolets containing the info (when number of iolets less than 30)

		/*
		// Approach 1
		// Read the Iolet info (iolet ids and fluid sites range) from GMem_Iolets_info
		site_t *Iolet_info = new site_t[3*num_local_Iolets];
		for (int index = 0; index< (3*num_local_Iolets); index++)
		{
			Iolet_info[index] = GMem_Iolets_info[index];
		}

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = Iolet_info[0]; //Iolets_info.Iolets_ID_range[0];// IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)
			// TODO: Replace this: _determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet); // _determine_Iolet_ID(num_local_Iolets, iolets_ID_range, Ind, &IdInlet);
			_determine_Iolet_ID(num_local_Iolets, Iolet_info, Ind, &IdInlet);
		}
		*/
		//
		// Approach 2 (Should be Faster - consider testing this):
		// Access the info from the GPU's constant memory: _Iolets_Inlet_Inner[local_iolets_MaxSIZE], local_iolets_MaxSIZE = 90 cuda_params.h (Assume 30 max iolets per RANK)
		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = Iolets_info.Iolets_ID_range[0];// IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)
			_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet); // _determine_Iolet_ID(num_local_Iolets, iolets_ID_range, Ind, &IdInlet);
		}
		//


		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
		/*else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}*/
		//--------------------------------------------------------------------------

		//==========================================================================
		// III. Load the Iolet-Fluid links info
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// Here is the loop over the LB lattice directions
#pragma unroll 19
			for (int LB_Dir = 1; LB_Dir < _NUMVECTORS; LB_Dir++) // keep the loop from LB_Dir=1
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				distribn_t correction = 0.0;
				distribn_t max_vel=0.0; // Max Velocity to be read from the velocityTable(IdInlet,t)

				int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
				// int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

				// Check which links are fluid-iolet links
				if(is_Iolet_link){
					//printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

					// A. Step: Load the prefactor correction term
					//	1. TODO
					site_t index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
					distribn_t prefactor_correction = GMem_dbl_wallMom_prefactor_correction[index_wallMom_correction];

					// Note that the prefactor contains the info (see Eval_wallMom_prefactor_correction in LaddIoletDelegate)
					/**
							LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));

							distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
                			* (wallMom_prefactor.x * LatticeType::CX[ii] + wallMom_prefactor.y * LatticeType::CY[ii]
                    		+ wallMom_prefactor.z * LatticeType::CZ[ii]) / Cs2;
					*/

					// Just multiply with max Velocity(IdInlet,t) from velocityTable
					// 	Load max Vel
					max_vel = GMem_Inlet_velocityTable[IdInlet *(total_TimeSteps+1) + time_Step]; // index_inlet*(total_TimeSteps+1)+timeStep

					// B. Step: Evaluate the single correction term as
					correction = prefactor_correction * max_vel;


						/*wallMom_x = inletNormal_x * vel_weight * max_vel;
						wallMom_y = inletNormal_y * vel_weight * max_vel;
						wallMom_z = inletNormal_z * vel_weight * max_vel;

						//------------------------------------------------------------------
						// C. Step: Evaluate the single correction term as
						correction = 2. * _EQMWEIGHTS_19[LB_Dir]
				                * (	wallMom_x * (double)_CX_19[LB_Dir] +
														wallMom_y * (double)_CY_19[LB_Dir] +
				                    wallMom_z * (double)_CZ_19[LB_Dir]) / _Cs2;
						*/

						// if (time_Step==1000 && correction!=0 && Ind==10065) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);

						// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
						// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
						// 	*** Do that inside the collision-streaming kernels Instead ***
						//		correction *= nn;
						//==================================================================

						//
						// Save the correction term in GPU global memory
						// TODO: Need to pass the:
						//		siteCount and shifted_Fluid_Ind
						// siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
						// shifted_Fluid_Ind is
						//int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

						//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
						//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));

						GMem_dbl_WallMom[index_wallMom_correction] = correction;
						//
				} // Closes the loop if(is_Iolet_link)

				//if (time_Step==1 && correction!=0)
					//printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);


			} // ends the loop over the LB_Dir directions
			//==========================================================================

} // Ends the GPU kernel

//**************************************************************
/* April 2023
		New kernel for evaluating the wall momentum corection terms
			on the GPU.

		Approach:
			A. Use the geometric prefactor associated with the wall momentum
					correction terms.
					See ...
			B. Just multiply with the maximum velocity(t) for the Corresponding
						iolet from the velocity table

		Identify the iolet index from the fluid index and info passed as an argument to the GPU kernel
			Iolets Iolets_info containing:
		iolet ID and fluid sites range

		Consider moving the above information in GPU constant memory
*/
//**************************************************************
//**************************************************************
__global__ void GPU_WallMom_correction_File_prefactor_NoIoletIDSearch(
											distribn_t* GMem_dbl_wallMom_prefactor_correction,
											distribn_t* GMem_dbl_WallMom,
											uint32_t* GMem_uint32_Iolet_Link,
											int IdInlet,
											distribn_t* GMem_Inlet_velocityTable,
											site_t start_Fluid_ID_givenColStreamType, site_t site_Count_givenColStreamType,
											site_t lower_limit, site_t upper_limit,
											unsigned long time_Step, unsigned long total_TimeSteps, unsigned long start_time)
{
	unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;

	Ind = Ind + lower_limit;

	if(Ind >= upper_limit)
		return;

		/** Load the following info:
					I. 		Velocity table Information - Requires iolet ID
					II. 	local iolet ID provided in the GPU kernel's arguments now (to be used in the velocity table)
					III. 	Iolet link info (which direction has a link to an iolet)
					IV. 	WallMom prefactor correction term
		*/

		//==========================================================================
		// III. Load the Iolet-Fluid links info
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// Here is the loop over the LB lattice directions
#pragma unroll 19
			for (int LB_Dir = 1; LB_Dir < _NUMVECTORS; LB_Dir++) // keep the loop from LB_Dir=1
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				// Check which links are fluid-iolet links
				if(is_Iolet_link){
					//printf("Enters the loop with an iolet link - Fluid ID: %lld, LB_Dir = %d \n", Ind, LB_Dir);

					//distribn_t correction = 0.0;
					//distribn_t max_vel=0.0; // Max Velocity to be read from the velocityTable(IdInlet,t)

					int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;
					// int index = shifted_Fluid_Ind*(_NUMVECTORS-1) + LB_Dir-1;

					// A. Step: Load the prefactor correction term
					site_t index_wallMom_correction = (LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind;
					distribn_t prefactor_correction = GMem_dbl_wallMom_prefactor_correction[index_wallMom_correction];

					// Note that the prefactor contains the info (see Eval_wallMom_prefactor_correction in LaddIoletDelegate)
					/**
							LatticeVelocity wallMom_prefactor(iolet->GetVelocity_prefactor(halfWay, bValues->GetTimeStep()));

							distribn_t prefactor_correction = 2. * LatticeType::EQMWEIGHTS[ii]
                			* (wallMom_prefactor.x * LatticeType::CX[ii] + wallMom_prefactor.y * LatticeType::CY[ii]
                    		+ wallMom_prefactor.z * LatticeType::CZ[ii]) / Cs2;
					*/

					// Just multiply with max Velocity(IdInlet,t) from velocityTable
					// 	Load max Vel
					distribn_t max_vel = GMem_Inlet_velocityTable[IdInlet *(total_TimeSteps+1) + time_Step - start_time]; // index_inlet*(total_TimeSteps+1)+timeStep

					// B. Step: Evaluate the single correction term as
					distribn_t correction = prefactor_correction * max_vel;

					// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
					// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
					// 	*** Do that inside the collision-streaming kernels Instead ***
					//		correction *= nn;
					//==================================================================

						//
						// Save the correction term in GPU global memory
						// TODO: Need to pass the:
						//		siteCount and shifted_Fluid_Ind
						// siteCount is site_Count_Inlet_Inner, site_Count_InletWall_Inner etc
						// shifted_Fluid_Ind is
						//int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

						//if (time_Step==1 && correction!=0) printf("GPU - Fluid ID: %lld, LB-Dir: %d, vel_weight: %f, wallMom_x: %.5e, wallMom_y: %.5e, wallMom_z: %.5e, correction: %.5e \n", Ind, LB_Dir, vel_weight, wallMom_x, wallMom_y, wallMom_z, correction);
						//if(time_Step==1 && correction!=0) printf("GPU - Fluid shifted ID: %lld, Index WallMom: %lld \n", shifted_Fluid_Ind, ((unsigned long long)(LB_Dir - 1) * site_Count_givenColStreamType + shifted_Fluid_Ind));

						GMem_dbl_WallMom[index_wallMom_correction] = correction;
						//
				} // Closes the loop if(is_Iolet_link)

				//if (time_Step==1 && correction!=0)
					//printf("GPU - Fluid ID: %lld, LB-Dir: %d, correction: %5.3e \n", Ind, LB_Dir, correction);

			} // ends the loop over the LB_Dir directions
			//==========================================================================

} // Ends the GPU kernel


__global__ void GPU_Check_Coordinates(int64_t *GMem_Coords_iolets,
																	site_t start_Fluid_ID_givenColStreamType,
																	site_t lower_limit, site_t upper_limit
																	)
{
	//unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
	//Ind = Ind + lower_limit;

	//if(Ind >= upper_limit)
		//return;
		printf("Enter GPU_Check_Coordinates kernel... \n" );
		for (int64_t Ind = lower_limit; Ind <= upper_limit; Ind++ ) {// TODO: Check the limits

			// 1. Load the coordinates of the point for which we would like tp evaluate the correction terms
			// Have in mind that (save registers per thread):
			int64_t shifted_Fluid_Ind = Ind - start_Fluid_ID_givenColStreamType;

			// Address is Misaligned (threads #0 to #31)
			int64_t x_coord = GMem_Coords_iolets[shifted_Fluid_Ind*3];
			int64_t y_coord = GMem_Coords_iolets[shifted_Fluid_Ind*3 + 1];
			int64_t z_coord = GMem_Coords_iolets[shifted_Fluid_Ind*3 + 2];

			//printf("Inside GPU kernel - Fluid Index = %lld, start_Fluid_ID_givenColStreamType = %lld, Shifted Index = %lld \n", Ind, start_Fluid_ID_givenColStreamType, shifted_Fluid_Ind);
			printf("Test coords kernel - Fluid Index = %lld, Shifted Index = %lld, Coordinates: (x, y, z) = (%lld, %lld, %lld) \n", Ind, shifted_Fluid_Ind, x_coord, y_coord, z_coord);

		}
}



/**
		Test kernel for the Velocity BCs case
		Check if the velocity table and the
		weights_table are correctly send on the GPU
*/
	__global__ void GPU_Check_Velocity_BCs_table_weights(int64_t **GMem_pp_int_weightsTable_coord,
																	distribn_t **GMem_pp_dbl_weightsTable_wei,
																	int inlet_ID,
																	distribn_t* GMem_Inlet_velocityTable,
																	int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets)
	/*__global__ void GPU_Check_Velocity_BCs_table_weights(int **GMem_pp_int_weightsTable_coord, int inlet_ID,
																												distribn_t* GMem_Inlet_velocityTable,
																												int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets
																											)*/
	{
		//unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		/*Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;
		*/

		printf("From GPU kernel: n_Elements: %d in weights_table from inlet ID: %d \n\n", n_arr_elementsInCurrentInlet_weightsTable, inlet_ID);

		//printf("Addresses...\n");
  	//printf("GMem_pp_int_weightsTable_coord[%d] = %p\n", inlet_ID, GMem_pp_int_weightsTable_coord[0]);

		for (int ii=0; ii<n_arr_elementsInCurrentInlet_weightsTable; ii++)
		//for (int ii=0; ii<1; ii++)
		{
			/*printf("Coordinates (x,y,z) : (%d, %d, %d ) - Weight : %f \n",  GMem_pp_int_weightsTable_coord[inlet_ID][ii*3],
																												GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+1],
																												GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2],
																												GMem_pp_dbl_weightsTable_wei[inlet_ID][ii]	);
			*/
			int64_t x_coord = GMem_pp_int_weightsTable_coord[inlet_ID][ii*3];
			int64_t y_coord = GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+1];
			int64_t z_coord = GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2];

			distribn_t vel_weight = GMem_pp_dbl_weightsTable_wei[inlet_ID][ii];

			//printf("GPU - Coordinates (x,y,z) : (%lld, %lld, %lld ) - Weight: %f \n", x_coord, y_coord, z_coord, vel_weight);
			//printf("Velocity Table : %f \n", GMem_Inlet_velocityTable[ii]);
		}

	}


	__global__ void GPU_Check_Velocity_BCs_table_weights_directArr(int *GMem_p_int_weightsTable_coord, int inlet_ID,
																												distribn_t* GMem_Inlet_velocityTable,
																												int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets
																											)
	{
		//unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		/*Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;
		*/

		/*for (int index = 0; index < arr_elementsInEachInlet.size(); index++) {
			//if (data[n][i] != 1) printf("kernel error\n");
		}*/
		printf("From GPU kernel: n_Elements: %d in weights_table from inlet ID: %d \n\n", n_arr_elementsInCurrentInlet_weightsTable, inlet_ID);
		//printf("Inside the GPU kernel ... \n");

		printf("Addresses...\n");
  	//printf("dX    = %p\n", d_X);
  	//printf("dA    = %p\n", d_A);
  	//printf("dB    = %p\n", d_B);
  	printf("GMem_p_int_weightsTable_coord = %p\n", GMem_p_int_weightsTable_coord[0]);
  	//printf("GMem_pp_int_weightsTable_coord[1] = %p\n", GMem_pp_int_weightsTable_coord[1]);

		//for (int ii=0; ii<n_arr_elementsInCurrentInlet_weightsTable; ii++)
		for (int ii=0; ii<2; ii++)
		{
			/*printf("Coordinates (x,y,z) : (%d, %d, %d ) - Weight : %f \n",  GMem_pp_int_weightsTable_coord[inlet_ID][ii*3],
																												GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+1],
																												GMem_pp_int_weightsTable_coord[inlet_ID][ii*3+2],
																												GMem_pp_dbl_weightsTable_wei[inlet_ID][ii]	);
																												*/
			int x_coord = GMem_p_int_weightsTable_coord[ii*3];
			int y_coord = GMem_p_int_weightsTable_coord[ii*3+1];
			int z_coord = GMem_p_int_weightsTable_coord[ii*3+2];

			// double vel_weight =
			printf("NEW KERNEL - GPU - Coordinates (x,y,z) : (%d, %d, %d ) \n", x_coord, y_coord, z_coord);
		}

		//int x_coord =
		//std::cout << "x_coord = " << GMem_p_int_weightsTable_coord[0] << std::endl;

	}


	__global__ void GPU_Check_Velocity_BCs_table_weights_directArr_v2(int *GMem_p_int_weightsTable_coord_x,
																												int *GMem_p_int_weightsTable_coord_y,
																												int *GMem_p_int_weightsTable_coord_z,
																												int inlet_ID,
																												distribn_t* GMem_Inlet_velocityTable,
																												int n_arr_elementsInCurrentInlet_weightsTable, int n_Inlets
																											)
	{

		//for (int ii=0; ii<n_arr_elementsInCurrentInlet_weightsTable; ii++)
		for (int ii=0; ii<2; ii++)
		{
			int x_coord = GMem_p_int_weightsTable_coord_x[ii];
			int y_coord = GMem_p_int_weightsTable_coord_y[ii];
			int z_coord = GMem_p_int_weightsTable_coord_z[ii];

			// double vel_weight =
			printf("NEW KERNEL(2) - GPU - Coordinates (x,y,z) : (%d, %d, %d ) \n", x_coord, y_coord, z_coord);
		}

	}



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
	//
	// April 2023 - Add the evaluation of the wall shear stress magnitude
	// 	Load:
	//		a. Wall normals
	//**************************************************************
	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress(distribn_t* GMem_dbl_fOld_b,
										distribn_t* GMem_dbl_fNew_b,
										distribn_t* GMem_dbl_MacroVars,
										site_t* GMem_int64_Neigh,
										uint32_t* GMem_uint32_Wall_Link,
										site_t nArr_dbl,
										site_t lower_limit_MidFluid, site_t upper_limit_MidFluid,
										site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, bool write_GlobalMem,
										distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
										unsigned long time_Step, int MPI_Rank)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit_MidFluid;

		if(Ind >= upper_limit_Wall)
			return;

		//printf("lower_limit_MidFluid: %lld, upper_limit_MidFluid: %lld, lower_limit_Wall: %lld, upper_limit_Wall: %lld \n\n", lower_limit_MidFluid, upper_limit_MidFluid, lower_limit_Wall, upper_limit_Wall);


		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19]; //, dev_fEq[19];
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

		/*// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;
		velz = momentum_z * density_1;
		*/
		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions
		double density_1 = 1.0 / nn;
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		double f_neq[19];
		if(write_GlobalMem){
#pragma unroll 19
			for (int i = 0; i < _NUMVECTORS; ++i)
			{
				double mom_dot_ei = (double)_CX_19[i] * momentum_x
												+ (double)_CY_19[i] * momentum_y
												+ (double)_CZ_19[i] * momentum_z;

				/*
						dev_fEq[i] = _EQMWEIGHTS_19[i]
										* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
														+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
			 */

			  double dev_fEq = _EQMWEIGHTS_19[i]
										* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
														+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

				f_neq[i] = dev_ff[i] - dev_fEq;
				dev_ff[i] += f_neq[i] * dev_minusInvTau;
			}
		}
		else{
			#pragma unroll 19
			for (int i = 0; i < _NUMVECTORS; ++i)
			{
				double mom_dot_ei = (double)_CX_19[i] * momentum_x
					+ (double)_CY_19[i] * momentum_y
					+ (double)_CZ_19[i] * momentum_z;

				/*
					dev_fEq[i] = _EQMWEIGHTS_19[i]
						* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
						+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				*/

				double dev_fEq = _EQMWEIGHTS_19[i]
													* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

				//f_neq[i] = dev_ff[i] - dev_fEq;
				dev_ff[i] += (dev_ff[i] - dev_fEq) * dev_minusInvTau;
			}
		}


		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements

		/*
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau;
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices

		// b. If within the limits for the mWallCollision
		//		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes
		//		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)

		site_t index_wall = nArr_dbl * _NUMVECTORS; // typedef int64_t site_t;

		GMem_dbl_fNew_b[Ind]= dev_ff[0];

#pragma unroll 18
		for(int LB_Dir=1; LB_Dir< _NUMVECTORS; LB_Dir++){
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
		//if (time_Step%_Send_MacroVars_DtH ==0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;

			velx = momentum_x * density_1;
			vely = momentum_y * density_1;
			velz = momentum_z * density_1;

			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;


			/*printf("Site: % ld, MidFluid limits: [%ld, %ld), Wall limits: [%ld, %ld) \n", Ind,
						lower_limit_MidFluid, upper_limit_MidFluid,
						lower_limit_Wall, upper_limit_Wall);
						*/
			//------------------------------------------------------------------------
			// Add here an if wallShearStressMagn_Eval as well
			// Evaluate the wall shear stress magnitude if this is a wall site
			// The first approach should be faster (Is it ?)

			//if (((site_t)Ind - upper_limit_Wall +1) * ((site_t)Ind - lower_limit_Wall) <= 0){		// When the upper_limit is NOT included
			if( (Ind >= lower_limit_Wall) && (Ind < upper_limit_Wall) ){
					distribn_t stress;

					//printf("Site: % ld, MidFluid limits: [%ld, %ld), Wall limits: [%ld, %ld) \n", Ind,
					//			lower_limit_MidFluid, upper_limit_MidFluid,
					//			lower_limit_Wall, upper_limit_Wall);

					// Load the wall normal components from the GPU global memory
					site_t shifted_Ind = Ind-lower_limit_Wall;
					distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
					distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
					distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
					//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

					stress = _CalculateWallShearStressMagnitude(nn,
						f_neq,
						wall_normal_x, wall_normal_y, wall_normal_z,
						_iStressParameter);

					//stress=0.001;

					/*if(shifted_Ind==9099 && MPI_Rank==206)
						printf("Rank: %d, Time: %ld, Site: %ld, upper_limit_MidFluid: %ld, upper_limit_Wall: %ld, Shifted Index: %ld, Wall normal components: (%5.5e, %5.5e, %5.5e), stress: %5.5e\n", MPI_Rank, time_Step, Ind, upper_limit_MidFluid, upper_limit_Wall, shifted_Ind, wall_normal_x, wall_normal_y, wall_normal_z, stress);
					*/
					//if(shifted_Ind==9099)
					//		printf("(1) Shifted Index = %ld,  Wall Shear Stress = %5.5e, _iStressParameter = %5.5e \n",shifted_Ind, stress, _iStressParameter );

					GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			}
			//------------------------------------------------------------------------
		} // Ends the loop if (write_GlobalMem)

	} // Ends the merged kernels GPU_Collide Types 1 & 2: mMidFluidCollision & mWallCollision
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
	//
	// April 2023 - Add the evaluation of the wall shear stress magnitude
	// 	Load:
	//		a. Wall normals
	//
	// August 2024 - Sponge Layer - LES formulation
	//	Load:
	//		a. the local vTau term (associated with the evaluation of the local relaxation time)
	//		b. LIfetime of the sponge layer (in case it dissolves after a certain time)
	// 		c. The time-step is also needed
	//**************************************************************
	__global__ void GPU_CollideStream_mMidFluidCollision_mWallCollision_sBB_WallShearStress(distribn_t* GMem_dbl_fOld_b,
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
									)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit_MidFluid;

		if(Ind >= upper_limit_Wall)
			return;

		//printf("lower_limit_MidFluid: %lld, upper_limit_MidFluid: %lld, lower_limit_Wall: %lld, upper_limit_Wall: %lld \n\n", lower_limit_MidFluid, upper_limit_MidFluid, lower_limit_Wall, upper_limit_Wall);

		// Sponge Layer - LES formulation
		// Load the local vTau value
		double _vTau = GMem_dbl_vTau[Ind];
		//printf("GPU - value of vTau: %f \n", _vTau);

		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19]; //, dev_fEq[19];
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

		/*// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;
		velz = momentum_z * density_1;
		*/
		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions
		double density_1 = 1.0 / nn;
		double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

		double f_neq[19];
		#pragma unroll 19
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
												+ (double)_CY_19[i] * momentum_y
												+ (double)_CZ_19[i] * momentum_z;

			double dev_fEq = _EQMWEIGHTS_19[i]
										* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
														+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

			f_neq[i] = dev_ff[i] - dev_fEq;

			// Sponge Layer - LES formulation
			// Compute the local relaxation time
			// dev_tau is tau0
			distribn_t local_tau =  _CalculateTau(dev_tau, dev_smag_cnst, _vTau, time_Step, SL_lifetime, f_neq);
			//printf("Local LES tau: %f, dev_tau: %f, _vTau: %f, SL_lifetime: %ld \n", local_tau, dev_tau, _vTau, SL_lifetime);

			//dev_ff[i] += f_neq[i] * dev_minusInvTau;
			dev_ff[i] += f_neq[i] * (-1./local_tau);
		}



		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements

		/*
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau;
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices

		// b. If within the limits for the mWallCollision
		//		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes
		//		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)

		site_t index_wall = nArr_dbl * _NUMVECTORS; // typedef int64_t site_t;

		GMem_dbl_fNew_b[Ind]= dev_ff[0];

	#pragma unroll 18
		for(int LB_Dir=1; LB_Dir< _NUMVECTORS; LB_Dir++){
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
		//if (time_Step%_Send_MacroVars_DtH ==0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;

			velx = momentum_x * density_1;
			vely = momentum_y * density_1;
			velz = momentum_z * density_1;

			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;


			/*printf("Site: % ld, MidFluid limits: [%ld, %ld), Wall limits: [%ld, %ld) \n", Ind,
						lower_limit_MidFluid, upper_limit_MidFluid,
						lower_limit_Wall, upper_limit_Wall);
						*/
			//------------------------------------------------------------------------
			// Add here an if wallShearStressMagn_Eval as well
			// Evaluate the wall shear stress magnitude if this is a wall site
			// The first approach should be faster (Is it ?)

			//if (((site_t)Ind - upper_limit_Wall +1) * ((site_t)Ind - lower_limit_Wall) <= 0){		// When the upper_limit is NOT included
			if( (Ind >= lower_limit_Wall) && (Ind < upper_limit_Wall) ){
					distribn_t stress;

					//printf("Site: % ld, MidFluid limits: [%ld, %ld), Wall limits: [%ld, %ld) \n", Ind,
					//			lower_limit_MidFluid, upper_limit_MidFluid,
					//			lower_limit_Wall, upper_limit_Wall);

					// Load the wall normal components from the GPU global memory
					site_t shifted_Ind = Ind-lower_limit_Wall;
					distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
					distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
					distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
					//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

					stress = _CalculateWallShearStressMagnitude(nn,
						f_neq,
						wall_normal_x, wall_normal_y, wall_normal_z,
						_iStressParameter);

					//stress=0.001;

					/*if(shifted_Ind==9099 && MPI_Rank==206)
						printf("Rank: %d, Time: %ld, Site: %ld, upper_limit_MidFluid: %ld, upper_limit_Wall: %ld, Shifted Index: %ld, Wall normal components: (%5.5e, %5.5e, %5.5e), stress: %5.5e\n", MPI_Rank, time_Step, Ind, upper_limit_MidFluid, upper_limit_Wall, shifted_Ind, wall_normal_x, wall_normal_y, wall_normal_z, stress);
					*/
					//if(shifted_Ind==9099)
					//		printf("(1) Shifted Index = %ld,  Wall Shear Stress = %5.5e, _iStressParameter = %5.5e \n",shifted_Ind, stress, _iStressParameter );

					GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			}
			//------------------------------------------------------------------------
		} // Ends the loop if (write_GlobalMem)

	} // Ends the merged kernels GPU_Collide Types 1 & 2: mMidFluidCollision & mWallCollision
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
										site_t lower_limit_Wall, site_t upper_limit_Wall, site_t totalSharedFs, bool write_GlobalMem)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit_MidFluid;

		if(Ind >= upper_limit_Wall)
			return;

		//printf("lower_limit_MidFluid: %lld, upper_limit_MidFluid: %lld, lower_limit_Wall: %lld, upper_limit_Wall: %lld \n\n", lower_limit_MidFluid, upper_limit_MidFluid, lower_limit_Wall, upper_limit_Wall);


		// Load the distribution functions
		//f[19] and fEq[19]
		double dev_ff[19]; //, dev_fEq[19];
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

		/*// Compute velocity components
		velx = momentum_x * density_1;
		vely = momentum_y * density_1;
		velz = momentum_z * density_1;
		*/
		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions

		//double momentumMagnitudeSquared = momentum_x * momentum_x
		//											+ momentum_y * momentum_y + momentum_z * momentum_z;

		double f_neq[19];
#pragma unroll 19
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			/*
			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
			*/
			double dev_fEq = _EQMWEIGHTS_19[i]
							* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

			f_neq[i] = dev_ff[i] - dev_fEq;
			dev_ff[i] += (dev_ff[i] - dev_fEq) * dev_minusInvTau;
		}

		// Add here an if wallShearStressMagn_Eval
		// This will initially evaluate the second moments of the distr. functions
		// Explicitly calculate the elements (0,0) (1,0) (1,1) (2,0) (2,1) (2,2)
		// and saves these with this order in the array ret_SecMomDistrFunc
		// 	Need then to exploit symmetry to fill the elements (0,1) (0,2) (1,2)
		double *SecMomDistrFunc;
		SecMomDistrFunc = _CalculatePiTensor(f_neq);
		SecMomDistrFunc[6] = SecMomDistrFunc[1];
		SecMomDistrFunc[7] = SecMomDistrFunc[3];
		SecMomDistrFunc[8] = SecMomDistrFunc[4];
		/*// Debugging
		for (int i = 0; i < 6; ++i) {
			printf("Second Mom. Distr. funct. %5.5e\n", SecMomDistrFunc[i]);
		}*/
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements

		/*
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau;
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices

		// b. If within the limits for the mWallCollision
		//		LOAD the Wall-Fluid links info - Remember that this is done for all the fluid nNodes
		//		Memory allocation in the future must be restricted to just the fluid nodes next to walls (i.e. the siteCount involved)

		site_t index_wall = nArr_dbl * _NUMVECTORS; // typedef int64_t site_t;

		GMem_dbl_fNew_b[Ind]= dev_ff[0];

#pragma unroll 18
		for(int LB_Dir=1; LB_Dir< _NUMVECTORS; LB_Dir++){
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
		//if (time_Step%_Send_MacroVars_DtH ==0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;

			velx = momentum_x * density_1;
			vely = momentum_y * density_1;
			velz = momentum_z * density_1;

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
