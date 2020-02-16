
// This file is part of the GPU development for HemeLB
// 7-1-2019

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
	__constant__ unsigned int _NUMVECTORS;
	__constant__ double dev_tau;
	__constant__ double dev_minusInvTau;

	__constant__ int _InvDirections_19[19];
	
	__constant__ double _EQMWEIGHTS_19[19];
	
	__constant__ int _CX_19[19];
	__constant__ int _CY_19[19];
	__constant__ int _CZ_19[19];



	//===================================================================================================================
	// __global__ GPU kernels
	
	//**************************************************************
	// Kernel for the Collision step 
	// for the Lattice Boltzmann algorithm 
	// Collision Type 2: mWallCollision: Wall-Fluid interaction 
	// Implementation currently follows the memory arrangement of the data 
	// by index LB, i.e. method (b)
	// Need to pass the information for the wall-fluid links - Done!!!
	//**************************************************************
	__global__ void GPU_CollideStream_2(double* GMem_dbl_fOld_b, 
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
	
		for(int i=0; i< _NUMVECTORS; i++){
			dev_ff[i] = GMem_dbl_fOld_b[(unsigned long long)i * nArr_dbl + Ind];			
		}

		__syncthreads(); // Check if this is needed or maybe I can have the density calculation within the loop 


		//-----------------------------------------------------------------------------------------------------------
		// Calculate the nessessary elements for calculating the equilibrium distribution functions
		// a. Calculate density
		// b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){			
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

		// Compute velocity components
		velx = momentum_x/nn;
		vely = momentum_y/nn;
		velz = momentum_z/nn;		

		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions
		double density_1 = 1.0 / nn;
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
		double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future
		
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{	
			dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i]; 
		}

		__syncthreads(); // Check if needed!
		

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// a. Load the streaming indices
		// b. The Wall-Fluid links info 
		
		// a. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 
			
			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!		

		}		
		__syncthreads();
		

		//
		// b. Wall-Fluid links info: 
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];


		// Put the new populations after collision in the GMem_dbl array, 
		// implementing the streaming step with Simple Bounce Back if Wall-Fluid link
		
				// fNew (dev_fn) populations:								
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)																	 
		{		
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask); 
			
			if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//printf("Site ID = %lld - Wall in Dir: %d \n\n", Ind, LB_Dir);
				// Simple Bounce Back case: 								
				GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_fn[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir
			}
			else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//---------------------------------------------------------------------------
				// If it streams in direction inside the simulation domain then it will point to a fluid ID < nFluid_nodes, otherwise it will stream to a neighbouring rank (place in the totalSharedFs at the end of the array)
				if (dev_NeighInd[LB_Dir] < (nArr_dbl*_NUMVECTORS) ) // maximum Data Address in array that correspond to this domain = nFluid_nodes*_NUMVECTORS
				{				
					dev_NeighInd[LB_Dir] = (dev_NeighInd[LB_Dir] - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index  							
					
					// Save the post collision population in fNew
					GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd[LB_Dir]] = dev_fn[LB_Dir];							
				}
				else{
					// Save the post collision population in fNew[Addr] at the end of the array in the (1+totalSharedFs) location  
					GMem_dbl_fNew_b[dev_NeighInd[LB_Dir]] = dev_fn[LB_Dir];		
					
					//
					// Debugging - Remove later
					// Check if it points to an address outside the (nFluid_nodes * _NUMVECTORS + 1+totalSharedFs )
					if (dev_NeighInd[LB_Dir] >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, dev_NeighInd[LB_Dir] );
					//
				}

				//---------------------------------------------------------------------------
			}	
		
		}				

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

		// printf("Lower Limit = %d \n\n", lower_limit);		
		// printf("Upper Limit = %d \n\n", upper_limit);
		
		//printf("Device: Relaxation Time = %.5f \n\n", dev_tau);	
		//printf("Device: Minus Inv. Relaxation Time = %.5f \n\n", dev_minusInvTau);			

		/*
		// All information below has been successfully passed to GPU memory 
		printf("GPU kernel Index = %d \n\n", Ind );
		printf("Number of vectors = %d \n\n", _NUMVECTORS);	
	
		printf("Info for Inv Directions... \n\n");	
		for (int i=0; i<19; i++){
			printf("Inv_Direction[%d] = %d \n\n", i, _InvDirections_19[i]);	
		}				
		printf("Info for Eqm Weights... \n\n");	
		for (int i=0; i<19; i++){
			printf("_EQMWEIGHTS_19[%d] = %.5f \n\n", i, _EQMWEIGHTS_19[i]);	
		}
		*/
		// printf("Outside the loop - Number of vectors = %d \n\n", _NUMVECTORS);
	
		/*
		for (int i=0; i < _NUMVECTORS; i++){
		    printf("Outside: CX[%d] = %d \n\n", i, _CX_19[i]);	
		  }				
		*/
		/*
		if(Ind == lower_limit){
			printf("Inside the loop -Lower Limit = %d \n\n", lower_limit);		
		  printf("Inside the loop - Number of vectors = %d \n\n", _NUMVECTORS);
		  printf("Inside the loop - Info for Discrete Velocities CX... \n\n");	
		  
		  for (int i=0; i < _NUMVECTORS; i++){
		    printf("Inside: CX[%d] = %d \n\n", i, _CX_19[i]);	
		  }				
		} // Passed		
		*/

		// Load the distribution functions		
		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19]; 
		double nn = 0.0;	// density 
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;

		double velx, vely, velz;	// Fluid Velocity 
	
		for(int i=0; i< _NUMVECTORS; i++){
			dev_ff[i] = GMem_dbl_fOld_b[(unsigned long long)i * nArr_dbl + Ind];			
		}

		__syncthreads(); // Check if this is needed or maybe I can have the density calculation within the loop 


		//-----------------------------------------------------------------------------------------------------------
		// Calculate the nessessary elements for calculating the equilibrium distribution functions
		// a. Calculate density
		// b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){			
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

		// Compute velocity components
		velx = momentum_x/nn;
		vely = momentum_y/nn;
		velz = momentum_z/nn;		

		//-----------------------------------------------------------------------------------------------------------
		// c. Calculate equilibrium distr. functions
		double density_1 = 1.0 / nn;
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

		/*		
		// Print to check 	
		if(Ind == lower_limit){
			printf("Eq. Distr. Functions: fEq[0] = %.5f ,\n fEq[1] = %.5f,\n fEq[2] = %.5f,\n fEq[3] = %.5f,\n fEq[4] = %.5f,\n fEq[5] = %.5f,\n fEq[6] = %.5f,\n fEq[7] = %.5f,\n fEq[8] = %.5f,\n fEq[9] = %.5f,\n fEq[10] = %.5f,\n fEq[11] = %.5f \n\n", dev_fEq[0], dev_fEq[1], dev_fEq[2], dev_fEq[3], dev_fEq[4], dev_fEq[5], dev_fEq[6], dev_fEq[7], dev_fEq[8], dev_fEq[9], dev_fEq[10], dev_fEq[11]);
		}
		*/
	
				
		// Collision step:
		// Single Relaxation Time approximation (LBGK)				
		double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future
		
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{	
			dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i]; 
		}

		__syncthreads(); // Check if needed!
		

		// Streaming Step -Load the streaming indices
		// dev_NeighInd[19] refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 
			
			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
			
			/*
			//
			// Debugging
			dev_NeighInd[LB_Dir] = (GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind] - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index  							
			unsigned long long streaming_Addr = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];

			//if (dev_NeighInd[i] > nArr_dbl) printf("Fluid Index = %lld, Streamed Neigh.[%d] = %lld, Total Fluid sites = %lld, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, i, dev_NeighInd[i], nArr_dbl, nArr_dbl*_NUMVECTORS+1+totalSharedFs, (GMem_int64_Neigh[(unsigned long long)i * nArr_dbl + Ind]) );
			if(dev_NeighInd[LB_Dir]> nArr_dbl && streaming_Addr >= ((unsigned long long)_NUMVECTORS*nArr_dbl)) 
			{	
				if (streaming_Addr >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, (GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]) );
			}
			//
			*/					

		}		
		__syncthreads();
		
		
		// Write old density and velocity to memory - 
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		GMem_dbl_MacroVars[Ind] = nn;
		GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
		GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
		GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

				
		// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well		
		// fn populations:
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){			

			// If it streams in direction inside the simulation domain then it will point to a fluid ID < nFluid_nodes, otherwise it will stream to a neighbouring rank (place in the totalSharedFs at the end of the array)
			if (dev_NeighInd[LB_Dir] < (nArr_dbl*_NUMVECTORS) ) // maximum Data Address in array that correspond to this domain = nFluid_nodes*_NUMVECTORS
			{				
				dev_NeighInd[LB_Dir] = (dev_NeighInd[LB_Dir] - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index  							
				
				// Save the post collision population in fNew
				GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd[LB_Dir]] = dev_fn[LB_Dir];		
	
				//GMem_dbl_fNew_b[(unsigned long long)i * nArr_dbl + Ind] = dev_fn[i];	// No streaming - Just saves the post collision distribution value	
			}
			else{
				// Save the post collision population in fNew[Addr] at the end of the array in the (1+totalSharedFs) location  
				GMem_dbl_fNew_b[dev_NeighInd[LB_Dir]] = dev_fn[LB_Dir];		

				// Check if it points to an address outside the (nFluid_nodes * _NUMVECTORS + 1+totalSharedFs )
				if (dev_NeighInd[LB_Dir] >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, dev_NeighInd[LB_Dir] );
			}	
		}
		
	} // Ends the kernel GPU_Collide
	//==========================================================================================




	//========================================================
	// Kernel for the Collision step 
	// for the Lattice Boltzmann algorithm 
	// Collision Type 1: Mid Domain - All neighbours are Fluid nodes
	// Implementation currently follows the memory arrangement of the data 
	// by index LB, i.e. method (b)
	//========================================================
	__global__ void GPU_CollideStream_1_PreSend(double* GMem_dbl_fOld_b, 
												double* GMem_dbl_fNew_b, 
												int64_t* GMem_int64_Neigh, 
												uint64_t nArr_dbl, uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;	
		Ind =Ind + lower_limit;					
		
		// printf("Lower Limit = %d \n\n", lower_limit);		
		// printf("Upper Limit = %d \n\n", upper_limit);

		if(Ind >= upper_limit)
			return;
		
		//printf("Device: Relaxation Time = %.5f \n\n", dev_tau);	
		//printf("Device: Minus Inv. Relaxation Time = %.5f \n\n", dev_minusInvTau);			

		/*
		// All information below has been successfully passed to GPU memory 
		printf("GPU kernel Index = %d \n\n", Ind );
		printf("Number of vectors = %d \n\n", _NUMVECTORS);	
	
		printf("Info for Inv Directions... \n\n");	
		for (int i=0; i<19; i++){
			printf("Inv_Direction[%d] = %d \n\n", i, _InvDirections_19[i]);	
		}				
		printf("Info for Eqm Weights... \n\n");	
		for (int i=0; i<19; i++){
			printf("_EQMWEIGHTS_19[%d] = %.5f \n\n", i, _EQMWEIGHTS_19[i]);	
		}
		*/
		// printf("Outside the loop - Number of vectors = %d \n\n", _NUMVECTORS);
	
		/*
		for (int i=0; i < _NUMVECTORS; i++){
		    printf("Outside: CX[%d] = %d \n\n", i, _CX_19[i]);	
		  }				
		*/
		/*
		if(Ind == lower_limit){
			printf("Inside the loop -Lower Limit = %d \n\n", lower_limit);		
		  printf("Inside the loop - Number of vectors = %d \n\n", _NUMVECTORS);
		  printf("Inside the loop - Info for Discrete Velocities CX... \n\n");	
		  
		  for (int i=0; i < _NUMVECTORS; i++){
		    printf("Inside: CX[%d] = %d \n\n", i, _CX_19[i]);	
		  }				
		} // Passed		
		*/

		// Load the distribution functions		
		//f[19] and fEq[19]
		double dev_ff[19], dev_fEq[19]; 
		double nn = 0.0;	// density 
		double momentum_x, momentum_y, momentum_z;
		momentum_x = momentum_y = momentum_z = 0.0;

		double velx, vely, velz;	// Fluid Velocity 
	
		for(int i=0; i< _NUMVECTORS; i++){
			dev_ff[i] = GMem_dbl_fOld_b[(unsigned long long)i * nArr_dbl + Ind];			
		}

		__syncthreads(); // Check if this is needed or maybe I can have the density calculation within the loop 

		// Calculate the nessessary elements for calculating the equilibrium distribution functions
		// a. Calculate density
		// b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		for(int direction = 0; direction< _NUMVECTORS; direction++){			
			nn += dev_ff[direction];
			momentum_x += _CX_19[direction] * dev_ff[direction];
			momentum_y += _CY_19[direction] * dev_ff[direction];
			momentum_z += _CZ_19[direction] * dev_ff[direction];
		}

		/*
		// In the case of body force 
		momentum_x += 0.5 * _force_x;
		momentum_y += 0.5 * _force_y;
		momentum_z += 0.5 * _force_z;
		*/		

		// Compute velocity components
		velx = momentum_x/nn;
		vely = momentum_y/nn;
		velz = momentum_z/nn;
		

		// c. Calculate equilibrium distr. functions
		const double density_1 = 1. / nn;
		const double momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;
		
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			const double mom_dot_ei = _CX_19[i] * momentum_x + _CY_19[i] * momentum_y
									+ _CZ_19[i] * momentum_z;

			dev_fEq[i] = _EQMWEIGHTS_19[i]
							* (nn - (3. / 2.) * momentumMagnitudeSquared * density_1
											+ (9. / 2.) * density_1 * mom_dot_ei * mom_dot_ei + 3. * mom_dot_ei);
		}
		
		/*		
		// Print to check 	
		if(Ind == lower_limit){
			printf("Eq. Distr. Functions: fEq[0] = %.5f ,\n fEq[1] = %.5f,\n fEq[2] = %.5f,\n fEq[3] = %.5f,\n fEq[4] = %.5f,\n fEq[5] = %.5f,\n fEq[6] = %.5f,\n fEq[7] = %.5f,\n fEq[8] = %.5f,\n fEq[9] = %.5f,\n fEq[10] = %.5f,\n fEq[11] = %.5f \n\n", dev_fEq[0], dev_fEq[1], dev_fEq[2], dev_fEq[3], dev_fEq[4], dev_fEq[5], dev_fEq[6], dev_fEq[7], dev_fEq[8], dev_fEq[9], dev_fEq[10], dev_fEq[11]);
		}
		*/
	
				
		// Collision step:
		// Single Relaxation Time approximation (LBGK)				
		double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future
		
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{	
			dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i]; 
		}

		__syncthreads(); // Check if needed!
		

		// Streaming Step -Load the streaming indices
		// dev_NeighInd[19] refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 
			
			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Refers to Data Address NOT THE STREAMING FLUID ID!!!
			
			/*
			//
			// Debugging
			dev_NeighInd[LB_Dir] = (GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind] - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index  							
			unsigned long long streaming_Addr = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];

			//if (dev_NeighInd[i] > nArr_dbl) printf("Fluid Index = %lld, Streamed Neigh.[%d] = %lld, Total Fluid sites = %lld, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, i, dev_NeighInd[i], nArr_dbl, nArr_dbl*_NUMVECTORS+1+totalSharedFs, (GMem_int64_Neigh[(unsigned long long)i * nArr_dbl + Ind]) );
			if(dev_NeighInd[LB_Dir]> nArr_dbl && streaming_Addr >= ((unsigned long long)_NUMVECTORS*nArr_dbl)) 
			{	
				if (streaming_Addr >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, (GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]) );
			}
			//
			*/					

		}		
		__syncthreads();

						
		// Put the new populations after collision in the GMem_dbl array, implementing the streaming step as well		
		// fn populations:
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){			

			// If it streams in direction inside the simulation domain then it will point to a fluid ID < nFluid_nodes, otherwise it will stream to a neighbouring rank (place in the totalSharedFs at the end of the array)
			if (dev_NeighInd[LB_Dir] < (nArr_dbl*_NUMVECTORS) ) // maximum Data Address in array that correspond to this domain = nFluid_nodes*_NUMVECTORS
			{				
				dev_NeighInd[LB_Dir] = (dev_NeighInd[LB_Dir] - LB_Dir)/_NUMVECTORS;	// Evaluate the ACTUAL streaming fluid ID index  							
				
				// Save the post collision population in fNew
				GMem_dbl_fNew_b[(unsigned long long)LB_Dir * nArr_dbl + dev_NeighInd[LB_Dir]] = dev_fn[LB_Dir];			
				//GMem_dbl_fNew_b[(unsigned long long)i * nArr_dbl + Ind] = dev_fn[i];	// No streaming - Just saves the post collision distribution value	
			}
			else{
				// Save the post collision population in fNew[Addr] at the end of the array in the (1+totalSharedFs) location  
				GMem_dbl_fNew_b[dev_NeighInd[LB_Dir]] = dev_fn[LB_Dir];		

				// Check if it points to an address outside the (nFluid_nodes * _NUMVECTORS + 1+totalSharedFs )
				if (dev_NeighInd[LB_Dir] >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, dev_NeighInd[LB_Dir] );
			}	
		}

		
	} // Ends the kernel GPU_Collide
	//==========================================================================================



	//==========================================================================================
	__global__ void GPUCalcMacroVars(double* GMem_dbl_fOld_b, double* GMem_dbl_fNew_b, unsigned int nArr_dbl, long long lower_limit, long long upper_limit)
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

			/*
			// In the case of body force 
			momentum_x += 0.5 * _force_x;
			momentum_y += 0.5 * _force_y;
			momentum_z += 0.5 * _force_z;
			*/		

			//Fluid velocity
			double u[3];
			/*
			u[0] = (F[1] - F[2] + (F[7]  - F[8]  + F[9]  - F[10] + F[15] - F[16] + F[17] - F[18]));
			u[1] = (F[3] - F[4] + (F[7]  + F[8]  - F[9]  - F[10] + F[11] - F[12] + F[13] - F[14]));
			u[2] = (F[5] - F[6] + (F[11] + F[12] - F[13] - F[14] + F[15] + F[16] - F[17] - F[18]));

			//Calc new velocity

			for(int i=0; i<3; i++)
				u[i] = (u[i] + _dtF_2[i]) / Density;
			*/

			__syncthreads();
			
			/*
			//Write new density and velocity to memory
			GMem[38ULL*nArr + Ind] = Density;

			for(int i=0; i<3; i++)
				GMem[(39ULL + i)*nArr + Ind] = u[i];				//u[i][NodeInd]
			*/

		}

#endif
}
