
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

	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm 
	// Collision Type 3: mInletCollision: Inlet BCs
	//
	// Inlet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
 	//	Two Possible types of Inlet BCs: 
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)	 
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)	 
	//
	// Implementation currently follows the memory arrangement of the data 
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - To do!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	// 
	//**************************************************************
	__global__ void GPU_CollideStream_3_NashZerothOrderPressure(double* GMem_dbl_fOld_b, 
																double* GMem_dbl_fNew_b, 
																double* GMem_dbl_MacroVars, 
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Iolet_Link, 										
																double* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,										
																uint64_t nArr_dbl, 
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;	
		Ind = Ind + lower_limit;					
		
		if(Ind >= upper_limit)
			return;
	/*
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

		
		// In the case of body force 
		//momentum_x += 0.5 * _force_x;
		//momentum_y += 0.5 * _force_y;
		//momentum_z += 0.5 * _force_z;
				

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
		// b. The Iolet-Fluid links info 
		// c. The ghost density 
		// d. The inletNormal


		// a. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory
		
		// printf("Number of inlets: %d \n\n", nInlets);
		double *ghost_dens = new double[nInlets];	// c. The ghost density		
		float *inletNormal = new float[3*nInlets];	// d. The inletNormal
	
	
		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind] 
			
			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!		

		}		

		__syncthreads();
		
		//
		// b. Iolet-Fluid links info: 
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];				


		ghost_dens[0] = GMem_ghostDensity[0];

		inletNormal[0] = GMem_inletNormal[0];
		inletNormal[1] = GMem_inletNormal[1];
		inletNormal[2] = GMem_inletNormal[2];

		// printf("ghost_dens[0]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", ghost_dens[0], inletNormal[0], inletNormal[1], inletNormal[2]);
		
		// Read the ghost density and the inlet Normal
		// How do I distinguish which inlet ID do I have ??? Need to think about this... To do!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)
	
		for (int IdInlet=0; IdInlet<nInlets; IdInlet++) {
			ghost_dens[IdInlet] = GMem_ghostDensity[IdInlet];
			//printf("IdInlet: %d, ghost_dens[%d]: %.5f \n\n", IdInlet, IdInlet, ghost_dens[IdInlet]);
			
			inletNormal[3*IdInlet] = GMem_inletNormal[3*IdInlet];
			inletNormal[3*IdInlet+1] = GMem_inletNormal[3*IdInlet+1];
			inletNormal[3*IdInlet+2] = GMem_inletNormal[3*IdInlet+2];			
		}	
	
		
		__syncthreads();

		// Put the new populations after collision in the GMem_dbl array, 
		// implementing the streaming step with Simple Bounce Back if Wall-Fluid link
		
		// fNew (dev_fn) populations:								
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)																	 
		{		
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask); 
			
			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				
				double component = velx*inletNormal[0] + vely*inletNormal[1] + velz*inletNormal[2];	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);
				
				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal[0] * component * ghost_dens[0];    
				double momentum_y = inletNormal[1] * component * ghost_dens[0];
				double momentum_z = inletNormal[2] * component * ghost_dens[0];
				
				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens[0];
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x 
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens[0] - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------	

				printf("Site ID = %lld - Inlet in Dir: %d \n\n", Ind, LB_Dir);
				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				// GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;				
				
			}
			else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!		
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
		*/
	} // Ends the kernel GPU_Collide Type 2: mWallCollision: Case Fluid-Wall collision
	//==========================================================================================




#endif // #ifdef HEMELB_USE_GPU
} // namespace hemelb
