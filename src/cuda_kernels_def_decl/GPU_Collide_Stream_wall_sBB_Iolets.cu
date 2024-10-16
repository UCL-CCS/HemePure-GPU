// This file is part of the GPU development for HemeLB
// 7-1-2019
/**
	Contains the GPU cuda kernels for the Iolet && Wall type of collision-streaming,
	i.e. for the InletWall and OutletWall collision-streaming
	and the 2 types of iolet BCs:
	 	1. Velocity BCs (LADDIOLET option in CMake file)
		2. Pressure BCs (NASHZEROTHORDERPRESSUREIOLET option in CMake file)

	Wall BCs: Simple Bounce Back
*/



#include <stdio.h>
#include "units.h"

#ifdef HEMELB_USE_GPU
#include "cuda_kernels_def_decl/cuda_params.h"
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
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
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

		//__syncthreads(); // Check if this is needed or maybe I can have the density calculation within the loop


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

		// __syncthreads(); // Check if needed!


		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of inlets: %d \n\n", nInlets);
		// How do I distinguish which inlet ID do I have ??? Need to think about this... To do!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)
		// Find a way to pass the IdInlet - To do!!!
		// Place that here:
		int IdInlet=0; // This will be replaced by whatever way we manage to read the inlet ID based on maybe fluid ID

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		//__syncthreads();


		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask;  // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Wall in Dir: %d \n\n", Ind, LB_Dir);
				// Simple Bounce Back case:
				GMem_dbl_fNew_b[(unsigned long long)_InvDirections_19[LB_Dir] * nArr_dbl + Ind]= dev_fn[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir

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

	} // Ends the kernel GPU_Collide Type 5-6: Case Iolets-Wall
	//==========================================================================================



	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Collision Type 5-6: Iolets-Wall BCs
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
 	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_new(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets, site_t* iolets_ID_range)
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
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, iolets_ID_range, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%100 == 0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 5-6: Case Iolets-Wall
	//==========================================================================================


	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Collision Type 5: Inlets-Wall BCs - PreSend
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
 	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Inlet_Edge(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets)
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
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = _Iolets_InletWall_Edge[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, _Iolets_InletWall_Edge, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%100 == 0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 5: Inlets-Wall - PreSend
	//==========================================================================================

	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Collision Type 5: Inlets-Wall BCs - PreReceive
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
 	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Inlet_Inner(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets)
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
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = _Iolets_InletWall_Inner[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, _Iolets_InletWall_Inner, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%100 == 0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 5: Inlets-Wall - PreReceive
	//==========================================================================================


	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Collision Type 6: Outlets-Wall BCs - PreSend
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
 	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Outlet_Edge(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets)
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
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = _Iolets_OutletWall_Edge[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, _Iolets_OutletWall_Edge, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%100 == 0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreSend
	//==========================================================================================



	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Collision Type 6: Outlets-Wall BCs - PreReceive
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
 	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_Outlet_Inner(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, int time_Step, int num_local_Iolets)
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
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = _Iolets_OutletWall_Inner[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, _Iolets_OutletWall_Inner, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		if(time_Step%100 == 0){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
	//==========================================================================================


	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
	// Collision Type 5, 6: Inlets/Outlets-Wall BCs
	//											Pass a struct to the kernel containing the Iolet info
	//												struct Iolets Iolets_info
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

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
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
#pragma unroll 19
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

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
#pragma unroll 19
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

			double dev_fEq = _EQMWEIGHTS_19[i]
													* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

		 	dev_ff[i] += (dev_ff[i] - dev_fEq) * dev_minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future
		/*
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = Iolets_info.Iolets_ID_range[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
#pragma unroll 19
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
	//==========================================================================================


	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
	// Collision Type 5, 6: Inlets/Outlets-Wall BCs
	//												Iolet's info accessed from GPU global memory
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_v2(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

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
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
#pragma unroll 19
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

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

#pragma unroll 19
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)_CX_19[i] * momentum_x
									+ (double)_CY_19[i] * momentum_y
									+ (double)_CZ_19[i] * momentum_z;

		  double dev_fEq = _EQMWEIGHTS_19[i]
													* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

			dev_ff[i] += (dev_ff[i] - dev_fEq) * dev_minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		/*// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links info, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		//--------------------------------------------------------------------------
		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// Read the Iolet info (iolet ids and fluid sites range) from GMem_Iolets_info
		//	extern __shared__ int s[];
		// TODO: Consider using shared memory in the future as this info is read from all the threads

		// Identify the local iolet ID
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

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

	/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
	*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
#pragma unroll 19
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}

	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
	//==========================================================================================




	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Velocity BCs: Option LADDIOLET
	// Collision Type 5, 6: Inlets/Outlets-Wall BCs
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//
	// Note regarding the wall mom:
	// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
	// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																uint64_t nArr_dbl,
																distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, bool write_GlobalMem)
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
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. Load the wallMom array
		// a.3. Compute the correction to the bounced back part of the distr. functions:
		//			Hence, need to have the following:
		// 			a.3.1. LatticeType::EQMWEIGHTS[LB_dir]
		//			a.3.2. LatticeType::CX[LB_dir], LatticeType::CY[LB_dir], LatticeType::CZ[LB_dir],
		//			a.3.3. Cs2
		//			a.3.4. Bounced-back index: just the INVERSEDIRECTIONS is sufficient
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

	/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
	*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			 unsigned mask = (LB_Dir > 0 ) ? 1U << (LB_Dir - 1 ) : 0;// Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = mask; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);

			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//=============================================================================================================
				// c. Load the WallMom info - Note: We follow Method b for the data layout
				site_t siteCount = upper_limit-lower_limit;
				site_t shifted_Fluid_Ind = Ind - lower_limit;
				//site_t nArr_wallMom = siteCount * (_NUMVECTORS-1); // Number of elements of type distribn_t(double)

				/*
				//-----------------------
				// Approach 1: Wall momentum passed to the GPU global memory (3 components: x,y,z)
				// Need to evaluate the correction term on the GPU (maybe this can be avoided - see approach 2)
				distribn_t WallMom_x, WallMom_y, WallMom_z;

				WallMom_x = GMem_dbl_WallMom[(unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];
				WallMom_y = GMem_dbl_WallMom[1ULL*nArr_wallMom + (unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];
				WallMom_z = GMem_dbl_WallMom[2ULL*nArr_wallMom + (unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

				//-----------------------
				// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
				// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
				WallMom_x *= nn;
				WallMom_y *= nn;
				WallMom_z *= nn;
				//-----------------------

				distribn_t correction = 2. * _EQMWEIGHTS_19[LB_Dir]
				                * (WallMom_x * _CX_19[LB_Dir] + WallMom_y * _CY_19[LB_Dir] + WallMom_z * _CZ_19[LB_Dir]) / _Cs2;
				//-----------------------
				*/

				//-----------------------
				// Approach 2
			 	// July 2022 - Single value correction term (wall momentum) passed to the GPU global memory
			 	distribn_t correction = GMem_dbl_WallMom[(unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

			 // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
			 // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
			 correction *= nn;
			 //-----------------------

				int unstreamed_dir = _InvDirections_19[LB_Dir];

				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_ff[LB_Dir] - correction;

				/*
				// Implement the following:
				// 	Iolet is in the LB direction = LB_dir
				* (latticeData->GetFNew(SimpleBounceBackDelegate<CollisionImpl>::GetBBIndex(site.GetIndex(), LB_dir))) =
				                hydroVars.GetFPostCollision()[LB_dir] - correction;
				*/
				//=============================================================================================================

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		//printf("_Send_MacroVars_DtH: %d \n\n", _Send_MacroVars_DtH);
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
	//==========================================================================================


	//**************************************************************
		// Kernel for the Collision step for the Lattice Boltzmann algorithm
		// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
		// Collision Type 5, 6: Inlets/Outlets-Wall BCs
		//											Pass a struct to the kernel containing the Iolet info
		//												struct Iolets Iolets_info
		//
		// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
		//	Two Possible types of Inlet BCs:
		// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
		//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
		//
		// Implementation currently follows the memory arrangement of the data
		// by index LB, i.e. method (b)
		// Need to pass the information for the fluid-iolet links - Done!!!
		// This information is in ioletIntersection, see geometry/SiteDataBare.h
		//
		//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
		//
		// April 2023 - Evaluate the wall shear stress magnitude - TODO
		//
		// August 2024 - Sponge Layer - LES formulation
		//	Load:
		//		a. the local vTau term (associated with the evaluation of the local relaxation time)
		//		b. LIfetime of the sponge layer (in case it dissolves after a certain time)
		// 		c. The time-step is also needed
		//**************************************************************
		__global__
		void GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress(
			distribn_t* GMem_dbl_fOld_b,
			distribn_t* GMem_dbl_fNew_b,
			distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh,
			uint32_t* GMem_uint32_Wall_Link,
			uint32_t* GMem_uint32_Iolet_Link,
			distribn_t* GMem_ghostDensity,
			float* GMem_inletNormal,
			int nInlets,
			uint64_t nArr_dbl,
			uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
			bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info,
			distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
			unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
		)
		{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind = Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

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
			// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		#pragma unroll 19
			for(int direction = 0; direction< _NUMVECTORS; direction++){
				dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

				nn += dev_ff[direction];
				momentum_x += (double)_CX_19[direction] * dev_ff[direction];
				momentum_y += (double)_CY_19[direction] * dev_ff[direction];
				momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
				//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
			}

			// Compute velocity components
			velx = momentum_x/nn;
			vely = momentum_y/nn;
			velz = momentum_z/nn;

			//-----------------------------------------------------------------------------------------------------------
			// c. Calculate equilibrium distr. functions
			double density_1 = 1.0 / nn;
			double momentumMagnitudeSquared = momentum_x * momentum_x
														+ momentum_y * momentum_y + momentum_z * momentum_z;

 			//
			// Sponge Layer - LES formulation
			// Load the local vTau value
			double _vTau = GMem_dbl_vTau[Ind];
			//printf("GPU - value of vTau: %f \n", _vTau);

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
			//-----------------------------------------------------------------------------------------------------------

			// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
			// To do!!!
			//-----------------------------------------------------------------------------------------------------------

			// Collision step:
			// Single Relaxation Time approximation (LBGK)
			//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future
			/*
			// Evolution equation for the fi's here
			for (int i = 0; i < _NUMVECTORS; ++i)
			{
				//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
				dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
			}
			*/

			// --------------------------------------------------------------------------------
			// Streaming Step:
			// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
			// Hence, Load the following:
			// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
			// a.2. ghost density
			// a.3. iolet normals
			// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
			// c. the bulk streaming indices

			//__syncthreads(); // Check if needed!

			//
			// a.1. Iolet-Fluid links info:
			uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

			// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
			distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
			float inletNormal_x, inletNormal_y, inletNormal_z;

			// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
			// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
			// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

			// Determine the IdInlet - Done!!!
			int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
			if(num_local_Iolets==1){
				IdInlet = Iolets_info.Iolets_ID_range[0];//IdInlet = iolets_ID_range[0];
			}
			else{
				// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
				// iolets_ID_range Array:
				//	a. Size: num_local_Iolets * 3
				// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
				_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
			}

			// Testing:
			if(IdInlet==INT32_MAX)
			{
				printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
			}
		/*		else{
				printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
			}
		*/

			ghost_dens = GMem_ghostDensity[IdInlet];
			inletNormal_x = GMem_inletNormal[3*IdInlet];
			inletNormal_y = GMem_inletNormal[3*IdInlet+1];
			inletNormal_z = GMem_inletNormal[3*IdInlet+2];
			//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

			// b. Wall-Fluid links info:
			uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		/*
			//------------------------------------------
			// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
			int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

			for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
				// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
			}
			//------------------------------------------
			__syncthreads();
		*/

			// Put the new populations after collision in the GMem_dbl array,
			// implementing the streaming step with:
			// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
			// Wall BCs: Simple Bounce Back if wall-fluid link

			// fNew (dev_fn) populations:
		#pragma unroll 19
			for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				unsigned mask_w = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Wall_link = (Wall_Intersect & mask_w);


				if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

					//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
					double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

					// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
					double momentum_x = inletNormal_x * component * ghost_dens;
					double momentum_y = inletNormal_y * component * ghost_dens;
					double momentum_z = inletNormal_z * component * ghost_dens;

					//------------------------------------------------------------------------------------------------------
					// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
					density_1 = 1.0 / ghost_dens;
					momentumMagnitudeSquared = momentum_x * momentum_x
														+ momentum_y * momentum_y + momentum_z * momentum_z;

					int unstreamed_dir = _InvDirections_19[LB_Dir];
					double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
										+ (double)_CY_19[unstreamed_dir] * momentum_y
										+ (double)_CZ_19[unstreamed_dir] * momentum_z;

					double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
								* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
												+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
					//------------------------------------------------------------------------------------------------------
					// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
					//=============================================================================================================

					// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

					// Case of NashZerothOrderPressure:
					// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
					GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

				}
				else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
				}

			}

			//=============================================================================================

			// Write old density and velocity to memory -
			// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
			// Check -  To do!!!
			//if(time_Step%_Send_MacroVars_DtH == 0){
			if (write_GlobalMem){
				GMem_dbl_MacroVars[Ind] = nn;
				GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
				GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
				GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

				//------------------------------------------------------------------------
				// Add here an if wallShearStressMagn_Eval as well
				// Evaluate the wall shear stress magnitude if this is a wall site
						distribn_t stress;

						// Load the wall normal components from the GPU global memory
						site_t shifted_Ind = Ind-lower_limit;
						distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
						distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
						distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
						//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

						stress = _CalculateWallShearStressMagnitude(nn,
							f_neq,
							wall_normal_x, wall_normal_y, wall_normal_z,
							_iStressParameter);

						//printf("(3) Wall Shear Stress = %5.5e\n", stress );
						GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
				//------------------------------------------------------------------------
			}
		} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive - Evaluating wall shear stress
		//==========================================================================================



//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
	// Collision Type 5, 6: Inlets/Outlets-Wall BCs
	//											Pass a struct to the kernel containing the Iolet info
	//												struct Iolets Iolets_info
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//
	// April 2023 - Evaluate the wall shear stress magnitude - TODO
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																bool write_GlobalMem, int num_local_Iolets, Iolets Iolets_info,
																distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

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
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
	#pragma unroll 19
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

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

		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future
		/*
		// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// printf("Number of iolets: %d \n\n", nIolets); // THis is the total number of iolets (whole geometry - NOT the local RANK iolets!!!)
		// How do I distinguish which inlet ID do I have ??? Need to think about this... Done!!!
		// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet = Iolets_info.Iolets_ID_range[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
		}

		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
	/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
	*/

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

	/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
	*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
	#pragma unroll 19
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

			//------------------------------------------------------------------------
			// Add here an if wallShearStressMagn_Eval as well
			// Evaluate the wall shear stress magnitude if this is a wall site
					distribn_t stress;

					// Load the wall normal components from the GPU global memory
					site_t shifted_Ind = Ind-lower_limit;
					distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
					distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
					distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
					//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

					stress = _CalculateWallShearStressMagnitude(nn,
						f_neq,
						wall_normal_x, wall_normal_y, wall_normal_z,
						_iStressParameter);

					//printf("(3) Wall Shear Stress = %5.5e\n", stress );
					GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			//------------------------------------------------------------------------
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive - Evaluating wall shear stress
	//==========================================================================================


	//**************************************************************
		// Kernel for the Collision step for the Lattice Boltzmann algorithm
		// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
		// Collision Type 5, 6: Inlets/Outlets-Wall BCs
		//												Iolet's info accessed from GPU global memory
		//
		// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
		//	Two Possible types of Inlet BCs:
		// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
		//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
		//
		// Implementation currently follows the memory arrangement of the data
		// by index LB, i.e. method (b)
		// Need to pass the information for the fluid-iolet links - Done!!!
		// This information is in ioletIntersection, see geometry/SiteDataBare.h
		//
		//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
		//
		// April 2023 - Evaluates the wall shear stress
		//
		// August 2024 - Sponge Layer - LES formulation
		//	Load:
		//		a. the local vTau term (associated with the evaluation of the local relaxation time)
		//		b. LIfetime of the sponge layer (in case it dissolves after a certain time)
		// 		c. The time-step is also needed
		//**************************************************************
		__global__
		void GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress(
			distribn_t* GMem_dbl_fOld_b,
			distribn_t* GMem_dbl_fNew_b,
			distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh,
			uint32_t* GMem_uint32_Wall_Link,
			uint32_t* GMem_uint32_Iolet_Link,
			distribn_t* GMem_ghostDensity,
			float* GMem_inletNormal,
			int nInlets,
			uint64_t nArr_dbl,
			uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
			bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info,
			distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
			unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime)
		{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind = Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

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
			// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
		#pragma unroll 19
			for(int direction = 0; direction< _NUMVECTORS; direction++){
				dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

				nn += dev_ff[direction];
				momentum_x += (double)_CX_19[direction] * dev_ff[direction];
				momentum_y += (double)_CY_19[direction] * dev_ff[direction];
				momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
				//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
			}

			// Compute velocity components
			velx = momentum_x/nn;
			vely = momentum_y/nn;
			velz = momentum_z/nn;

			//-----------------------------------------------------------------------------------------------------------
			// c. Calculate equilibrium distr. functions
			double density_1 = 1.0 / nn;
			double momentumMagnitudeSquared = momentum_x * momentum_x
														+ momentum_y * momentum_y + momentum_z * momentum_z;

			//
			// Sponge Layer - LES formulation
			// Load the local vTau value
			double _vTau = GMem_dbl_vTau[Ind];
			//printf("GPU - value of vTau: %f \n", _vTau);

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
			//-----------------------------------------------------------------------------------------------------------

			// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
			// To do!!!
			//-----------------------------------------------------------------------------------------------------------

			// Collision step:
			// Single Relaxation Time approximation (LBGK)
			//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

			/*// Evolution equation for the fi's here
			for (int i = 0; i < _NUMVECTORS; ++i)
			{
				//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
				dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
			}
			*/

			// --------------------------------------------------------------------------------
			// Streaming Step:
			// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
			// Hence, Load the following:
			// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
			// a.2. ghost density
			// a.3. iolet normals
			// b. the wall-fluid links info, i.e. GMem_uint32_Wall_Link
			// c. the bulk streaming indices

			//__syncthreads(); // Check if needed!

			//
			// a.1. Iolet-Fluid links info:
			uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

			//--------------------------------------------------------------------------
			// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
			distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
			float inletNormal_x, inletNormal_y, inletNormal_z;

			// Read the Iolet info (iolet ids and fluid sites range) from GMem_Iolets_info
			//	extern __shared__ int s[];
			// TODO: Consider using shared memory in the future as this info is read from all the threads

			// Identify the local iolet ID
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

			ghost_dens = GMem_ghostDensity[IdInlet];
			inletNormal_x = GMem_inletNormal[3*IdInlet];
			inletNormal_y = GMem_inletNormal[3*IdInlet+1];
			inletNormal_z = GMem_inletNormal[3*IdInlet+2];
			//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

			// b. Wall-Fluid links info:
			uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		/*
			//------------------------------------------
			// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
			int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

			for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
				// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
			}
			//------------------------------------------
			__syncthreads();
		*/

			// Put the new populations after collision in the GMem_dbl array,
			// implementing the streaming step with:
			// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
			// Wall BCs: Simple Bounce Back if wall-fluid link

			// fNew (dev_fn) populations:
		#pragma unroll 19
			for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				unsigned mask_w = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Wall_link = (Wall_Intersect & mask_w);


				if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

					//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
					double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

					// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
					double momentum_x = inletNormal_x * component * ghost_dens;
					double momentum_y = inletNormal_y * component * ghost_dens;
					double momentum_z = inletNormal_z * component * ghost_dens;

					//------------------------------------------------------------------------------------------------------
					// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
					density_1 = 1.0 / ghost_dens;
					momentumMagnitudeSquared = momentum_x * momentum_x
														+ momentum_y * momentum_y + momentum_z * momentum_z;

					int unstreamed_dir = _InvDirections_19[LB_Dir];
					double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
										+ (double)_CY_19[unstreamed_dir] * momentum_y
										+ (double)_CZ_19[unstreamed_dir] * momentum_z;

					double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
								* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
												+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
					//------------------------------------------------------------------------------------------------------
					// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
					//=============================================================================================================

					// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

					// Case of NashZerothOrderPressure:
					// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
					GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

				}
				else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
				}

			}

			//=============================================================================================

			// Write old density and velocity to memory -
			// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
			// Check -  To do!!!
			//if(time_Step%_Send_MacroVars_DtH == 0){
			if (write_GlobalMem){
				GMem_dbl_MacroVars[Ind] = nn;
				GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
				GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
				GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

				//------------------------------------------------------------------------
				// Add here an if wallShearStressMagn_Eval as well
				// Evaluate the wall shear stress magnitude if this is a wall site
				distribn_t stress;

				// Load the wall normal components from the GPU global memory
				site_t shifted_Ind = Ind-lower_limit;
				distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
				distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
				distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
				//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

				stress = _CalculateWallShearStressMagnitude(nn,
										f_neq,
										wall_normal_x, wall_normal_y, wall_normal_z,
										_iStressParameter);

				//printf("(4) Wall Shear Stress = %5.5e\n", stress );
				GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
				//------------------------------------------------------------------------
			}

		} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive - Evaluate the wall shear stress
		//==========================================================================================


//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
	// Collision Type 5, 6: Inlets/Outlets-Wall BCs
	//												Iolet's info accessed from GPU global memory
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//
	// April 2023 - Evaluates the wall shear stress - TODO!!!
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress(	distribn_t* GMem_dbl_fOld_b,
																distribn_t* GMem_dbl_fNew_b,
																distribn_t* GMem_dbl_MacroVars,
																int64_t* GMem_int64_Neigh,
																uint32_t* GMem_uint32_Wall_Link,
																uint32_t* GMem_uint32_Iolet_Link,
																distribn_t* GMem_ghostDensity,
																float* GMem_inletNormal,
																int nInlets,
																uint64_t nArr_dbl,
																uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs,
																bool write_GlobalMem, int num_local_Iolets, site_t* GMem_Iolets_info,
																distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

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
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
	#pragma unroll 19
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

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
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[_NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		/*// Evolution equation for the fi's here
		for (int i = 0; i < _NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * dev_minusInvTau; // Check if multiplying by dev_minusInvTau makes a difference
		}
		*/

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. ghost density
		// a.3. iolet normals
		// b. the wall-fluid links info, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//__syncthreads(); // Check if needed!

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		//--------------------------------------------------------------------------
		// a.2-3. Read the ghost density and the iolet (inlet/outlet) Normal (vector)
		distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
		float inletNormal_x, inletNormal_y, inletNormal_z;

		// Read the Iolet info (iolet ids and fluid sites range) from GMem_Iolets_info
		//	extern __shared__ int s[];
		// TODO: Consider using shared memory in the future as this info is read from all the threads

		// Identify the local iolet ID
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

		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

	/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
	*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
	#pragma unroll 19
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);


			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

				//printf("Site ID = %lld - Iolet in Dir: %d \n\n", Ind, LB_Dir);
				double component = velx*inletNormal_x + vely*inletNormal_y + velz*inletNormal_z;	// distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

				// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
				double momentum_x = inletNormal_x * component * ghost_dens;
				double momentum_y = inletNormal_y * component * ghost_dens;
				double momentum_z = inletNormal_z * component * ghost_dens;

				//------------------------------------------------------------------------------------------------------
				// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
				density_1 = 1.0 / ghost_dens;
				momentumMagnitudeSquared = momentum_x * momentum_x
													+ momentum_y * momentum_y + momentum_z * momentum_z;

				int unstreamed_dir = _InvDirections_19[LB_Dir];
				double mom_dot_ei = (double)_CX_19[unstreamed_dir] * momentum_x
									+ (double)_CY_19[unstreamed_dir] * momentum_y
									+ (double)_CZ_19[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = _EQMWEIGHTS_19[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * LatticeType::NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

			//------------------------------------------------------------------------
			// Add here an if wallShearStressMagn_Eval as well
			// Evaluate the wall shear stress magnitude if this is a wall site
			distribn_t stress;

			// Load the wall normal components from the GPU global memory
			site_t shifted_Ind = Ind-lower_limit;
			distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
			distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
			distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
			//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

			stress = _CalculateWallShearStressMagnitude(nn,
									f_neq,
									wall_normal_x, wall_normal_y, wall_normal_z,
									_iStressParameter);

			//printf("(4) Wall Shear Stress = %5.5e\n", stress );
			GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			//------------------------------------------------------------------------
		}

	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive - Evaluate the wall shear stress
	//==========================================================================================


	//**************************************************************
		// Kernel for the Collision step for the Lattice Boltzmann algorithm
		// Velocity BCs: Option LADDIOLET
		// Collision Type 5, 6: Inlets/Outlets-Wall BCs
		//
		// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
		//	Two Possible types of Inlet BCs:
		// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
		//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
		//
		// Implementation currently follows the memory arrangement of the data
		// by index LB, i.e. method (b)
		// Need to pass the information for the fluid-iolet links - Done!!!
		// This information is in ioletIntersection, see geometry/SiteDataBare.h
		//
		//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
		//
		// Note regarding the wall mom:
		// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
		// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:

		// August 2024 - Sponge Layer - LES formulation
		//	Load:
		//		a. the local vTau term (associated with the evaluation of the local relaxation time)
		//		b. LIfetime of the sponge layer (in case it dissolves after a certain time)
		// 		c. The time-step is also needed
		//**************************************************************
		__global__
		void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress(
				distribn_t* GMem_dbl_fOld_b,
				distribn_t* GMem_dbl_fNew_b,
				distribn_t* GMem_dbl_MacroVars,
				int64_t* GMem_int64_Neigh,
				uint32_t* GMem_uint32_Wall_Link,
				uint32_t* GMem_uint32_Iolet_Link,
				uint64_t nArr_dbl,
				distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom,
				uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, bool write_GlobalMem,
				distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal,
				unsigned long time_Step, distribn_t* GMem_dbl_vTau, unsigned long int SL_lifetime
			)
		{
			unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
			Ind = Ind + lower_limit;

			if(Ind >= upper_limit)
				return;

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
			// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
	#pragma unroll 19
			for(int direction = 0; direction< _NUMVECTORS; direction++){
				dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

				nn += dev_ff[direction];
				momentum_x += (double)_CX_19[direction] * dev_ff[direction];
				momentum_y += (double)_CY_19[direction] * dev_ff[direction];
				momentum_z += (double)_CZ_19[direction] * dev_ff[direction];
				//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
			}

			// Compute velocity components
			velx = momentum_x/nn;
			vely = momentum_y/nn;
			velz = momentum_z/nn;

			//-----------------------------------------------------------------------------------------------------------
			// c. Calculate equilibrium distr. functions
			double density_1 = 1.0 / nn;
			double momentumMagnitudeSquared = momentum_x * momentum_x
														+ momentum_y * momentum_y + momentum_z * momentum_z;

			// Sponge Layer - LES formulation
			// Load the local vTau value
			double _vTau = GMem_dbl_vTau[Ind];
			//printf("GPU - value of vTau: %f \n", _vTau);

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

			// --------------------------------------------------------------------------------
			// Streaming Step:
			// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
			// Hence, Load the following:
			// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
			// a.2. Load the wallMom array
			// a.3. Compute the correction to the bounced back part of the distr. functions:
			//			Hence, need to have the following:
			// 			a.3.1. LatticeType::EQMWEIGHTS[LB_dir]
			//			a.3.2. LatticeType::CX[LB_dir], LatticeType::CY[LB_dir], LatticeType::CZ[LB_dir],
			//			a.3.3. Cs2
			//			a.3.4. Bounced-back index: just the INVERSEDIRECTIONS is sufficient
			// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
			// c. the bulk streaming indices

			//
			// a.1. Iolet-Fluid links info:
			uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

			// b. Wall-Fluid links info:
			uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		/*
			//------------------------------------------
			// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
			int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

			for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
				// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
				// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
			}
			//------------------------------------------
			__syncthreads();
		*/

			// Put the new populations after collision in the GMem_dbl array,
			// implementing the streaming step with:
			// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
			// Wall BCs: Simple Bounce Back if wall-fluid link

			// fNew (dev_fn) populations:
	#pragma unroll 19
			for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
			{
				unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Iolet_link = (Iolet_Intersect & mask);

				unsigned mask_w = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
				bool is_Wall_link = (Wall_Intersect & mask_w);

				if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
					//=============================================================================================================
					// c. Load the WallMom info - Note: We follow Method b for the data layout
					site_t siteCount = upper_limit-lower_limit;
					site_t shifted_Fluid_Ind = Ind - lower_limit;
					//site_t nArr_wallMom = siteCount * (_NUMVECTORS-1); // Number of elements of type distribn_t(double)

					/*
					//-----------------------
					// Approach 1: Wall momentum passed to the GPU global memory (3 components: x,y,z)
					// Need to evaluate the correction term on the GPU (maybe this can be avoided - see approach 2)
					distribn_t WallMom_x, WallMom_y, WallMom_z;

					WallMom_x = GMem_dbl_WallMom[(unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];
					WallMom_y = GMem_dbl_WallMom[1ULL*nArr_wallMom + (unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];
					WallMom_z = GMem_dbl_WallMom[2ULL*nArr_wallMom + (unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

					//-----------------------
					// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
					// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
					WallMom_x *= nn;
					WallMom_y *= nn;
					WallMom_z *= nn;
					//-----------------------

					distribn_t correction = 2. * _EQMWEIGHTS_19[LB_Dir]
													* (WallMom_x * _CX_19[LB_Dir] + WallMom_y * _CY_19[LB_Dir] + WallMom_z * _CZ_19[LB_Dir]) / _Cs2;
					//-----------------------
					*/

					//-----------------------
					// Approach 2
					// July 2022 - Single value correction term (wall momentum) passed to the GPU global memory
					distribn_t correction = GMem_dbl_WallMom[(unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

				 // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
				 // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
				 correction *= nn;
				 //-----------------------

					int unstreamed_dir = _InvDirections_19[LB_Dir];

					GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_ff[LB_Dir] - correction;

					/*
					// Implement the following:
					// 	Iolet is in the LB direction = LB_dir
					* (latticeData->GetFNew(SimpleBounceBackDelegate<CollisionImpl>::GetBBIndex(site.GetIndex(), LB_dir))) =
													hydroVars.GetFPostCollision()[LB_dir] - correction;
					*/
					//=============================================================================================================

				}
				else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
				}

			}

			//=============================================================================================

			// Write old density and velocity to memory -
			// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
			// Check -  To do!!!
			//printf("_Send_MacroVars_DtH: %d \n\n", _Send_MacroVars_DtH);
			//if(time_Step%_Send_MacroVars_DtH == 0){
			if (write_GlobalMem){
				GMem_dbl_MacroVars[Ind] = nn;
				GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
				GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
				GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

				//------------------------------------------------------------------------
				// Add here an if wallShearStressMagn_Eval as well
				// Evaluate the wall shear stress magnitude if this is a wall site
				distribn_t stress;

				// Load the wall normal components from the GPU global memory
				site_t shifted_Ind = Ind-lower_limit;
				distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
				distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
				distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
				//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

				stress = _CalculateWallShearStressMagnitude(nn,
					f_neq,
					wall_normal_x, wall_normal_y, wall_normal_z,
					_iStressParameter);

				//printf("(3) Wall Shear Stress = %5.5e\n", stress );
				GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
				//------------------------------------------------------------------------
			}
		} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive - Evaluate Wall Shear Stress
		//==========================================================================================

//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// Velocity BCs: Option LADDIOLET
	// Collision Type 5, 6: Inlets/Outlets-Wall BCs
	//
	// Iolet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	//	This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//
	// Note regarding the wall mom:
	// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
	// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
	//**************************************************************
	__global__ void GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress(
			distribn_t* GMem_dbl_fOld_b,
			distribn_t* GMem_dbl_fNew_b,
			distribn_t* GMem_dbl_MacroVars,
			int64_t* GMem_int64_Neigh,
			uint32_t* GMem_uint32_Wall_Link,
			uint32_t* GMem_uint32_Iolet_Link,
			uint64_t nArr_dbl,
			distribn_t* GMem_dbl_WallMom, uint64_t nArr_wallMom,
			uint64_t lower_limit, uint64_t upper_limit, uint64_t totalSharedFs, bool write_GlobalMem,
			distribn_t* GMem_dbl_WallShearStressMagn, distribn_t* GMem_dbl_WallNormal)
	{
		unsigned long long Ind = blockIdx.x * blockDim.x + threadIdx.x;
		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

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
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
#pragma unroll 19
		for(int direction = 0; direction< _NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

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

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. Load the wallMom array
		// a.3. Compute the correction to the bounced back part of the distr. functions:
		//			Hence, need to have the following:
		// 			a.3.1. LatticeType::EQMWEIGHTS[LB_dir]
		//			a.3.2. LatticeType::CX[LB_dir], LatticeType::CY[LB_dir], LatticeType::CZ[LB_dir],
		//			a.3.3. Cs2
		//			a.3.4. Bounced-back index: just the INVERSEDIRECTIONS is sufficient
		// b. the wall-fluid links infor, i.e. GMem_uint32_Wall_Link
		// c. the bulk streaming indices

		//
		// a.1. Iolet-Fluid links info:
		uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

	/*
		//------------------------------------------
		// c. Bulk Streaming indices: dev_NeighInd[19] here refers to either: a) the ACTUAL fluid ID index or b) the hemeLB neighbourIndices which refer to the array Index (Data Address) in f_old and f_new
		int64_t dev_NeighInd[19]; // ACTUAL fluid ID index for the neighbours - or streaming Data Address in hemeLB f's memory

		for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){
			// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
			// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

			dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
		}
		//------------------------------------------
		__syncthreads();
	*/

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
#pragma unroll 19
		for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
		{
			unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);

			unsigned mask_w = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Wall_link = (Wall_Intersect & mask_w);

			if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//=============================================================================================================
				// c. Load the WallMom info - Note: We follow Method b for the data layout
				site_t siteCount = upper_limit-lower_limit;
				site_t shifted_Fluid_Ind = Ind - lower_limit;
				//site_t nArr_wallMom = siteCount * (_NUMVECTORS-1); // Number of elements of type distribn_t(double)

				/*
				//-----------------------
				// Approach 1: Wall momentum passed to the GPU global memory (3 components: x,y,z)
				// Need to evaluate the correction term on the GPU (maybe this can be avoided - see approach 2)
				distribn_t WallMom_x, WallMom_y, WallMom_z;

				WallMom_x = GMem_dbl_WallMom[(unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];
				WallMom_y = GMem_dbl_WallMom[1ULL*nArr_wallMom + (unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];
				WallMom_z = GMem_dbl_WallMom[2ULL*nArr_wallMom + (unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

				//-----------------------
				// TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
				// Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
				WallMom_x *= nn;
				WallMom_y *= nn;
				WallMom_z *= nn;
				//-----------------------

				distribn_t correction = 2. * _EQMWEIGHTS_19[LB_Dir]
												* (WallMom_x * _CX_19[LB_Dir] + WallMom_y * _CY_19[LB_Dir] + WallMom_z * _CZ_19[LB_Dir]) / _Cs2;
				//-----------------------
				*/

				//-----------------------
				// Approach 2
				// July 2022 - Single value correction term (wall momentum) passed to the GPU global memory
				distribn_t correction = GMem_dbl_WallMom[(unsigned long long)(LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

			 // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
			 // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
			 correction *= nn;
			 //-----------------------

				int unstreamed_dir = _InvDirections_19[LB_Dir];

				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_ff[LB_Dir] - correction;

				/*
				// Implement the following:
				// 	Iolet is in the LB direction = LB_dir
				* (latticeData->GetFNew(SimpleBounceBackDelegate<CollisionImpl>::GetBBIndex(site.GetIndex(), LB_dir))) =
												hydroVars.GetFPostCollision()[LB_dir] - correction;
				*/
				//=============================================================================================================

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
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
			}

		}

		//=============================================================================================

		// Write old density and velocity to memory -
		// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
		// Check -  To do!!!
		//printf("_Send_MacroVars_DtH: %d \n\n", _Send_MacroVars_DtH);
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;

			//------------------------------------------------------------------------
			// Add here an if wallShearStressMagn_Eval as well
			// Evaluate the wall shear stress magnitude if this is a wall site
			distribn_t stress;

			// Load the wall normal components from the GPU global memory
			site_t shifted_Ind = Ind-lower_limit;
			distribn_t wall_normal_x = GMem_dbl_WallNormal[3*shifted_Ind];
			distribn_t wall_normal_y = GMem_dbl_WallNormal[3*shifted_Ind + 1];
			distribn_t wall_normal_z = GMem_dbl_WallNormal[3*shifted_Ind + 2];
			//printf("Site: % ld, Wall normal components: (%5.5e, %5.5e, %5.5e)\n", Ind, wall_normal_x, wall_normal_y, wall_normal_z);

			stress = _CalculateWallShearStressMagnitude(nn,
				f_neq,
				wall_normal_x, wall_normal_y, wall_normal_z,
				_iStressParameter);

			//printf("(3) Wall Shear Stress = %5.5e\n", stress );
			GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			//------------------------------------------------------------------------
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive - Evaluate Wall Shear Stress
	//==========================================================================================


#endif // #ifdef HEMELB_USE_GPU
} // namespace hemelb
