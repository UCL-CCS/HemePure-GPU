#pragma once
#include <stdio.h>
#include "api.h"

namespace hemelb
{

	//==============================================================================
	/**
	  Device function to investigate which Iolet Ind corresponds to a fluid with index fluid_Ind
	  To be used for the inlet/outlet related collision-streaming kernels
	  Checks through the local iolets (inlet/outlet) to determine the correct iolet ID
	  each iolet has fluid sites with indices in the range: [lower_limit,upper_limit]
	  Function returns the iolet ID value: IdInlet.
	 */
	GPU_INLINE_DEVICE_FUNCTION void _determine_Iolet_ID(int num_local_Iolets, site_t* iolets_ID_range, site_t fluid_Ind, int* IdInlet)
	{
		// Loop over the number of local iolets (num_local_Iolets) and determine whether the fluid ID (fluid_Ind) falls whithin the range
		for (int i_local_iolet = 0; i_local_iolet<num_local_Iolets; i_local_iolet++)
		{
			// iolet range: [lower_limit,upper_limit)
			int64_t lower_limit = iolets_ID_range[3*i_local_iolet+1];	// Included in the fluids range
			int64_t upper_limit = iolets_ID_range[3*i_local_iolet+2];	// Value included in the fluids' range - CHANGED TO INCLUDE THE VALUE

			//if ((fluid_Ind - upper_limit +1) * (fluid_Ind - lower_limit) <= 0){	 	//When the upper_limit is NOT included
			if ((fluid_Ind - upper_limit) * (fluid_Ind - lower_limit) <= 0){ 				// When the upper_limit is included
				*IdInlet =(int)(iolets_ID_range[3*i_local_iolet]);
				return;
			}
		}// closes the loop over the local iolets
	}
	//==============================================================================


	//**************************************************************
	// Kernel for the Collision step for the Lattice Boltzmann algorithm
	// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
	// 	Collision Types 3-4: mInletCollision & mOutletCollision: Inlet - Outlet BCs
	//												Pass a struct to the kernel containing the Iolet info
	//													struct Iolets Iolets_info
	//
	// Inlet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
	//	Two Possible types of Inlet BCs:
	// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
	//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
	//
	// Implementation currently follows the memory arrangement of the data
	// 		by index LB, i.e. method (b)
	// Need to pass the information for the fluid-iolet links - Done!!!
	// 		This information is in ioletIntersection, see geometry/SiteDataBare.h
	//
	// This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
	//
	//**************************************************************
		template <typename LatticeType> struct GPU_CollideStream_Iolets_NashZerothOrderPressure_Functor {
			distribn_t *GMem_dbl_fOld_b;
			distribn_t *GMem_dbl_fNew_b;
			distribn_t *GMem_dbl_MacroVars;
			int64_t *GMem_int64_Neigh;
			uint32_t *GMem_uint32_Iolet_Link;
			distribn_t *GMem_ghostDensity;
			float *GMem_inletNormal;
			int nInlets;
			uint64_t nArr_dbl;
			uint64_t lower_limit;
			uint64_t upper_limit;
			uint64_t totalSharedFs;
			bool write_GlobalMem;
			int num_local_Iolets;
			Iolets Iolets_info;
			const double minusInvTau;

			const int myPiD;
			const int line;

			GPU_CollideStream_Iolets_NashZerothOrderPressure_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
					int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Iolet_Link_, distribn_t *GMem_ghostDensity_,
					float *GMem_inletNormal_, int nInlets_, uint64_t nArr_dbl_, uint64_t lower_limit_,
					uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_, int num_local_Iolets_,
					Iolets Iolets_info_, double minusInvTau_, int myPiD_, int line_)
				: GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
				GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), GMem_ghostDensity(GMem_ghostDensity_), GMem_inletNormal(GMem_inletNormal_), nInlets(nInlets_),
				nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_), totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_),
				num_local_Iolets(num_local_Iolets_), Iolets_info(Iolets_info_), minusInvTau(minusInvTau_), myPiD(myPiD_), line(line_) {}

			GPU_KERNEL void operator()(unsigned long long Ind) {
				const lattices::D3Q19GPUConstants c;

				Ind = Ind + lower_limit;

				if (Ind >= upper_limit)
					return;

				// Load the distribution functions
				// f[19] and fEq[19]
				double dev_ff[19];   //, dev_fEq[19];
				double nn = 0.0;     // density
				double momentum_x, momentum_y, momentum_z;
				momentum_x = momentum_y = momentum_z = 0.0;

				double velx, vely, velz;   // Fluid Velocity
										   //-------------------------------------------------------------------------------------------------------
										   // 1. Read the fOld_GPU_b distr. functions
										   // 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
										   // 		a. Calculate density
										   // 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
#pragma unroll 19
				for (int direction = 0; direction < c.NUMVECTORS; direction++) {
					dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long) direction * nArr_dbl + Ind];

					nn += dev_ff[direction];
					momentum_x += (double) c.CX[direction] * dev_ff[direction];
					momentum_y += (double) c.CY[direction] * dev_ff[direction];
					momentum_z += (double) c.CZ[direction] * dev_ff[direction];
				}

				// Compute velocity components
				velx = momentum_x / nn;
				vely = momentum_y / nn;
				velz = momentum_z / nn;

				//-------------------------------------------------------------------------------------------------------
				// c. Calculate equilibrium distr. functions
				double density_1 = 1.0 / nn;
				double momentumMagnitudeSquared = momentum_x * momentum_x + momentum_y * momentum_y 
					+ momentum_z * momentum_z;

#pragma unroll 19
				for (int i = 0; i < c.NUMVECTORS; ++i) {
					double mom_dot_ei = (double) c.CX[i] * momentum_x + (double) c.CY[i] * momentum_y 
						+ (double) c.CZ[i] * momentum_z;

					double dev_fEq = c.EQMWEIGHTS[i] * (nn - (3.0 / 2.0) * (momentum_x * momentum_x 
								+ momentum_y * momentum_y + momentum_z * momentum_z) * density_1 +
							(9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

					dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
				}
				//--------------------------------------------------------------------------------------------------
				// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
				// To do!!!
				//------------------------------------------------------------------------------------------------------

				// Collision step:
				// Single Relaxation Time approximation (LBGK)

				// --------------------------------------------------------------------------------
				// Streaming Step:
				// a. Load the streaming indices
				// b. The Iolet-Fluid links info
				// c. The ghost density
				// d. The inletNormal

				// a. Bulk Streaming indices: dev_NeighInd[19] here refers to the ACTUAL Streaming Array index (Data Address) in f_old and f_new

				distribn_t ghost_dens;   // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
				float inletNormal_x, inletNormal_y, inletNormal_z;

				// b. Iolet-Fluid links info:
				uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

				// Read the ghost density and the inlet Normal
				// How do I distinguish which inlet ID do I have ??? Need to think about this... To do!!!
				// Need to pass this info based on the site Index (from the initialisation process. With given site 
				//ranges -> int boundaryId = site.GetIoletId();)

				// Access the info from the constant memory: _Iolets_Inlet_Inner[local_iolets_MaxSIZE], 
				// local_iolets_MaxSIZE = 6 cuda_params.h (Assume 2 max iolets per
				// RANK) Determine the IdInlet - Done!!!
				int IdInlet = INT32_MAX;   // Iolet (Inlet/Outlet) ID

				if (num_local_Iolets ==  1) {
					IdInlet = (int) (Iolets_info.Iolets_ID_range[0]);   // IdInlet = iolets_ID_range[0];
				} else {
					_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind,
							&IdInlet);
				}
				if (IdInlet == INT32_MAX) {
					printf("Fluid_ID : %llu, ID_iolet: %d - Fluid NOT in IOLET range (NashZerothOrderPressure) !!! PID: %d file: lb.hpp line: %d  \n\n", Ind, IdInlet, myPiD, line);
				}

				ghost_dens = GMem_ghostDensity[IdInlet];
				inletNormal_x = GMem_inletNormal[3 * IdInlet];
				inletNormal_y = GMem_inletNormal[3 * IdInlet + 1];
				inletNormal_z = GMem_inletNormal[3 * IdInlet + 2];

				// Put the new populations after collision in the GMem_dbl array,
				// implementing the streaming step with Simple Bounce Back if Wall-Fluid link

				// fNew (dev_fn) populations:
#pragma unroll 19
				for (int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++) {
					unsigned mask = 1U << (LB_Dir - 1);   // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do:
														  // compare against test_bool_Wall_Intersect as well)
					bool is_Iolet_link = (Iolet_Intersect & mask);

					if (is_Iolet_link) {   // ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

						//===================================================================================================
						// Not valid in general! Need to change!!!
						// (IdInlet=0) Here we assume that we have only one inlet and the value of 
						// int boundaryId = site.GetIoletId() = 1. Need to change in the future!!!
						double component = velx * inletNormal_x + vely * inletNormal_y +
							velz * inletNormal_z;   // distribn_t component = (hydroVars.momentum / hydroVars.density).Dot(ioletNormal);

						// ghostHydrovars.momentum = ioletNormal * component * ghostDensity;
						double momentum_x = inletNormal_x * component * ghost_dens;
						double momentum_y = inletNormal_y * component * ghost_dens;
						double momentum_z = inletNormal_z * component * ghost_dens;

						//----------------------------------------------------------------------------------------------------
						// Calculate Feq[unstreamed_dir] - Only the direction that is necessary
						density_1 = 1.0 / ghost_dens;
						momentumMagnitudeSquared = momentum_x * momentum_x 
							+ momentum_y * momentum_y + momentum_z * momentum_z;

						int unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];
						double mom_dot_ei = (double) c.CX[unstreamed_dir] * momentum_x 
							+ (double) c.CY[unstreamed_dir] * momentum_y 
							+ (double) c.CZ[unstreamed_dir] * momentum_z;

						double dev_fEq_unstr = c.EQMWEIGHTS[unstreamed_dir] * (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1 +
								(9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
						//---------------------------------------------------------------------------------------------------
						// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the 
						// info (identify the proper ghost density and inlet-normals.
						//===================================================================================================

						// Case of NashZerothOrderPressure:
						GMem_dbl_fNew_b[(unsigned long long) unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

					} else {   // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

						// Use the Neighbouring Index given in GPUDataAddr_int64_Neigh_d, which is the actual 
						// streaming Array Index in f_new global memory
						int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long) LB_Dir * nArr_dbl + Ind];

						// Save the post collision population in fNew
						GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir];

						//---------------------------------------------------------------------------
					}   // Closes the bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				}
				//=============================================================================================

				// Write old density and velocity to memory -
				// Maybe use a different cuda kernel for these calculations 
				// (if saving the MacroVariables delays the collision/streaming kernel)
				// Check -  To do!!!
				// if(time_Step%_Send_MacroVars_DtH==0){
				if (write_GlobalMem) {
					GMem_dbl_MacroVars[Ind] = nn;
					GMem_dbl_MacroVars[1ULL * nArr_dbl + Ind] = velx;
					GMem_dbl_MacroVars[2ULL * nArr_dbl + Ind] = vely;
					GMem_dbl_MacroVars[3ULL * nArr_dbl + Ind] = velz;
				}

			}   // Ends the kernel GPU_Collide Type 4: mOutletCollision
				//==========================================================================================

			};   // End of functor


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

			__constant__ bool _useWeightsFromFile;


			__constant__ int _InvDirections_19[19];

			__device__ __constant__ double _EQMWEIGHTS_19[19];

			__constant__ int _CX_19[19];
			__constant__ int _CY_19[19];
			__constant__ int _CZ_19[19];

			__constant__ int _WriteStep = 100;
			__constant__ int _Send_MacroVars_DtH = 100; // Writing MacroVariables to GPU global memory (Sending MacroVariables calculated during the collision-streaming kernels to the GPU Global mem).



			//**************************************************************
			// Kernel for the Collision step for the Lattice Boltzmann algorithm
			// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
			// 	Collision Types 3-4: mInletCollision & mOutletCollision: Inlet - Outlet BCs
			//												Pass a struct to the kernel containing the Iolet info
			//													struct Iolets Iolets_info
			//
			// Inlet BCs: specified with HEMELB_INLET_BOUNDARY in CMakeLists.txt
			//	Two Possible types of Inlet BCs:
			// 	1. NashZerothOrderPressure: Implement this first (see lb/streamers/NashZerothOrderPressureDelegate.h)
			//	2. LaddIolet: (see lb/streamers/LaddIoletDelegate.h)
			//
			// Implementation currently follows the memory arrangement of the data
			// 		by index LB, i.e. method (b)
			// Need to pass the information for the fluid-iolet links - Done!!!
			// 		This information is in ioletIntersection, see geometry/SiteDataBare.h
			//
			// This version uses the ACTUAL streaming address in global memory - NOT the fluid ID.
			//
			//**************************************************************
			__global__ void GPU_CollideStream_Iolets_NashZerothOrderPressure(distribn_t* GMem_dbl_fOld_b,
					distribn_t* GMem_dbl_fNew_b,
					distribn_t* GMem_dbl_MacroVars,
					int64_t* GMem_int64_Neigh,
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
				double dev_ff[19]={0}; //, dev_fEq[19];
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
				// a. Load the streaming indices
				// b. The Iolet-Fluid links info
				// c. The ghost density
				// d. The inletNormal

				// a. Bulk Streaming indices: dev_NeighInd[19] here refers to the ACTUAL Streaming Array index (Data Address) in f_old and f_new
				//int64_t dev_NeighInd[19];

				// printf("Number of inlets: %d \n\n", nInlets);
				distribn_t ghost_dens; // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
				float inletNormal_x, inletNormal_y, inletNormal_z;

				/*
				   for(int LB_Dir=0; LB_Dir< _NUMVECTORS; LB_Dir++){

				// If we use the elements in GMem_int64_Neigh - then we access the memory address in fOld or fNew directly (not the fluid id)
				// (remember the memory layout in hemeLB is based on the site fluid index, i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				dev_NeighInd[LB_Dir] = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind]; // Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!
				}
				__syncthreads();
				 */
				//
				// b. Iolet-Fluid links info:
				uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];


				// Read the ghost density and the inlet Normal
				// How do I distinguish which inlet ID do I have ??? Need to think about this... To do!!!
				// Need to pass this info based on the site Index (from the initialisation process. With given site ranges -> int boundaryId = site.GetIoletId();)

				// Access the info from the constant memory: _Iolets_Inlet_Inner[local_iolets_MaxSIZE], local_iolets_MaxSIZE = 6 cuda_params.h (Assume 2 max iolets per RANK)
				// Determine the IdInlet - Done!!!

				int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
				if(num_local_Iolets==1){
					IdInlet =(int) Iolets_info.Iolets_ID_range[0];// IdInlet = iolets_ID_range[0];
				}
				else{
					// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
					// iolets_ID_range Array:
					//	a. Size: num_local_Iolets * 3
					// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit)
					_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet); // _determine_Iolet_ID(num_local_Iolets, iolets_ID_range, Ind, &IdInlet);
				}

				// Testing:
				if(IdInlet==INT32_MAX)
				{
					printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
				}

				//printf("Number of local Iolets = %d \n", num_local_Iolets);
				/*
				   for (int index = 0; index < num_local_Iolets; index++){
				   printf(" Iolet ID: %d, lower_range: %lld, upper_range: %lld ", _Iolets_Inlet_Inner[3*index], _Iolets_Inlet_Inner[3*index+1], _Iolets_Inlet_Inner[3*index+2]);
				   }
				   printf("\n\n");
				 */

				ghost_dens = GMem_ghostDensity[IdInlet];
				inletNormal_x = GMem_inletNormal[3*IdInlet];
				inletNormal_y = GMem_inletNormal[3*IdInlet+1];
				inletNormal_z = GMem_inletNormal[3*IdInlet+2];
				//		printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);


				// Put the new populations after collision in the GMem_dbl array,
				// implementing the streaming step with Simple Bounce Back if Wall-Fluid link

				// fNew (dev_fn) populations:
#pragma unroll 19
				for (int LB_Dir = 0; LB_Dir < _NUMVECTORS; LB_Dir++)
				{
					unsigned mask = 1U << (LB_Dir - 1); // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
					bool is_Iolet_link = (Iolet_Intersect & mask);

					if(is_Iolet_link){	// ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

						//=============================================================================================================
						// Not valid in general! Need to change!!!
						// (IdInlet=0) Here we assume that we have only one inlet and the value of int boundaryId = site.GetIoletId() = 1. Need to change in the future!!!
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
					else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

						// Use the Neighbouring Index given in GPUDataAddr_int64_Neigh_d, which is the actual streaming Array Index in f_new global memory
						int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long)LB_Dir * nArr_dbl + Ind];

						// Save the post collision population in fNew
						GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir];
						//
						// Debugging - Remove later
						// Check if it points to an address outside the (nFluid_nodes * _NUMVECTORS + 1+totalSharedFs )
						//if (dev_NeighInd[LB_Dir] >= (nArr_dbl*_NUMVECTORS+1+totalSharedFs)) printf("Error!!! Fluid Index = %lld, Stream.Dir.= %d, Max. Streaming addr = %lld Vs Stream. Addr.=%lld \n\n", Ind, LB_Dir, nArr_dbl*_NUMVECTORS+1+totalSharedFs, dev_NeighInd[LB_Dir] );
						//

						//---------------------------------------------------------------------------
					} // Closes the bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				}
				//=============================================================================================


				// Write old density and velocity to memory -
				// Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
				// Check -  To do!!!
				//if(time_Step%_Send_MacroVars_DtH==0){
				if (write_GlobalMem){
					GMem_dbl_MacroVars[Ind] = nn;
					GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
					GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
					GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
				}

			} // Ends the kernel GPU_Collide Type 4: mOutletCollision
			  //==========================================================================================


#include "./nash_data_1.h"

			template<typename LatticeType>
				bool setup()
				{
					bool initialise_GPU_res = true;
					// 2.a. Weight coefficients for the equilibrium distr. functions
					bool status = GPU::deviceMemcpyToSymbol(hemelb::_EQMWEIGHTS_19, LatticeType::EQMWEIGHTS, LatticeType::NUMVECTORS*sizeof(double), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (1)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
						//goto Error;
					}

					// 2.b. Number of vectors: LatticeType::NUMVECTORS
					static const unsigned int num_Vectors = LatticeType::NUMVECTORS;
					status = GPU::deviceMemcpyToSymbol(&hemelb::_NUMVECTORS, &num_Vectors, sizeof(num_Vectors), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (2)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
						//goto Error;
					}

					// 2.c. Inverse directions for the bounce back LatticeType::INVERSEDIRECTIONS[direction]
					status = GPU::deviceMemcpyToSymbol(hemelb::_InvDirections_19, LatticeType::INVERSEDIRECTIONS, LatticeType::NUMVECTORS*sizeof(int), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (3)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
						//goto Error;
					}

					// 2.d. Lattice Velocity directions CX[DmQn::NUMVECTORS], CY[DmQn::NUMVECTORS], CZ[DmQn::NUMVECTORS]
					status = GPU::deviceMemcpyToSymbol(hemelb::_CX_19, LatticeType::CX, LatticeType::NUMVECTORS*sizeof(int), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (4)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
					}
					status = GPU::deviceMemcpyToSymbol(hemelb::_CY_19, LatticeType::CY, LatticeType::NUMVECTORS*sizeof(int), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (5)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
					}
					status = GPU::deviceMemcpyToSymbol(hemelb::_CZ_19, LatticeType::CZ, LatticeType::NUMVECTORS*sizeof(int), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (6)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
					}

					status = GPU::deviceMemcpyToSymbol(&hemelb::dev_minusInvTau, &ParamData::minusInvTau, sizeof(&ParamData::minusInvTau), 0, GPU::memcpyHostToDevice);
					if (!status) {
						fprintf(stderr, "GPU constant memory copy failed (8)\n");
						initialise_GPU_res = false;
						return initialise_GPU_res;
						//return false;
					}

					return initialise_GPU_res;

				}
			} // End namespace 
