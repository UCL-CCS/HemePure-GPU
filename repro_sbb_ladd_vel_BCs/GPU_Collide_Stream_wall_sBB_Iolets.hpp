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
template <typename LatticeType> struct GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_Functor {
  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Wall_Link;
  uint32_t *GMem_uint32_Iolet_Link;
  uint64_t nArr_dbl;
  distribn_t *GMem_dbl_WallMom;
  uint64_t nArr_wallMom;
  uint64_t lower_limit;
  uint64_t upper_limit;
  uint64_t totalSharedFs;
  bool write_GlobalMem;
  const double minusInvTau;
  const double Cs2;

  GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
                                                        int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Wall_Link_, uint32_t *GMem_uint32_Iolet_Link_,
                                                        uint64_t nArr_dbl_, distribn_t *GMem_dbl_WallMom_, uint64_t nArr_wallMom_, uint64_t lower_limit_,
                                                        uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_, double minusInvTau_, double Cs2_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_), GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), nArr_dbl(nArr_dbl_),
        GMem_dbl_WallMom(GMem_dbl_WallMom_), nArr_wallMom(nArr_wallMom_), lower_limit(lower_limit_), upper_limit(upper_limit_), totalSharedFs(totalSharedFs_),
        write_GlobalMem(write_GlobalMem_), minusInvTau(minusInvTau_), Cs2(Cs2_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) {
	const lattices::D3Q19GPUConstants c;
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
		for(int direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
			//printf("Momentum: _x = %.5e, _y = %.5e, _z = %.5e \n\n", momentum_x, momentum_y, momentum_z);
		}
		__syncthreads();

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

		for (int i = 0; i < c.NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)c.CX[i] * momentum_x
					+ (double)c.CY[i] * momentum_y
					+ (double)c.CZ[i] * momentum_z;

			dev_fEq[i] = c.EQMWEIGHTS[i]
							* (nn - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[c.NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		// Evolution equation for the fi's here
		for (int i = 0; i < c.NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * minusInvTau; // Check if multiplying by minusInvTau makes a difference
		}

		// --------------------------------------------------------------------------------
		// Streaming Step:
		// HemeLB does things in the following order: (a) fluid-iolet, (b) fluid-wall and (c) fluid-fluid links.
		// Hence, Load the following:
		// a.1. the Iolet-Fluid links info, i.e. GMem_uint32_Iolet_Link
		// a.2. Load the wallMom array
		// a.3. Compute the correction to the bounced back part of the distr. functions:
		//			Hence, need to have the following:
		// 			a.3.1. LatticeType::c.EQMWEIGHTS[LB_dir]
		//			a.3.2. LatticeType::c.CX[LB_dir], LatticeType::c.CY[LB_dir], LatticeType::c.CZ[LB_dir],
		//			a.3.3. Cs2
		//			a.3.4. Bounced-back index: just the c.INVERSEDIRECTIONS is sufficient
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

		for(int LB_Dir=0; LB_Dir< c.NUMVECTORS; LB_Dir++){
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
		for (int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++)
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
				//site_t nArr_wallMom = siteCount * (c.NUMVECTORS-1); // Number of elements of type distribn_t(double)

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

				distribn_t correction = 2. * c.EQMWEIGHTS[LB_Dir]
				                * (WallMom_x * c.CX[LB_Dir] + WallMom_y * c.CY[LB_Dir] + WallMom_z * c.CZ[LB_Dir]) / Cs2;
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

				int unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];

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
				GMem_dbl_fNew_b[(unsigned long long)c.INVERSEDIRECTIONS[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir

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
  }   // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
      //==========================================================================================
};    // End functor



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


	__constant__ int _InvDirections_19[19];

	__device__ __constant__ double _EQMWEIGHTS_19[19];

	__constant__ int _CX_19[19];
	__constant__ int _CY_19[19];
	__constant__ int _CZ_19[19];

	__constant__ int _WriteStep = 100;
	__constant__ int _Send_MacroVars_DtH = 100; // Writing MacroVariables to GPU global memory (Sending MacroVariables calculated during the collision-streaming kernels to the GPU Global mem).




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
		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
	//==========================================================================================

} // namespace hemelb

#include "./precursor_data_7.h"

namespace hemelb { 

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

	status = GPU::deviceMemcpyToSymbol(&hemelb::_Cs2, &ParamData::Cs2, sizeof(ParamData::Cs2), 0, GPU::memcpyHostToDevice);
	if (!status) {
		fprintf(stderr, "GPU constant memory copy failed (9)\n");
		initialise_GPU_res = false;
		return initialise_GPU_res;
		//return false;
	}
	return initialise_GPU_res;

}
}; // end of namespace hemelb
