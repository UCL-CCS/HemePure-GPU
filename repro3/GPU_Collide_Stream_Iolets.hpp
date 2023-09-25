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



template <typename LatticeType> struct GPU_CollideStream_Iolets_Ladd_VelBCs_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Iolet_Link;
  uint64_t nArr_dbl;
  distribn_t *GMem_dbl_WallMom;
  uint64_t nArr_wallMom;
  uint64_t lower_limit;
  uint64_t upper_limit;
  uint64_t totalSharedFs;
  bool write_GlobalMem;

  const double minusInvTau;

  GPU_CollideStream_Iolets_Ladd_VelBCs_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
                                               int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Iolet_Link_, uint64_t nArr_dbl_, distribn_t *GMem_dbl_WallMom_,
                                               uint64_t nArr_wallMom_, uint64_t lower_limit_, uint64_t upper_limit_, uint64_t totalSharedFs_,
                                               bool write_GlobalMem_, double minusInvTau_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), nArr_dbl(nArr_dbl_), GMem_dbl_WallMom(GMem_dbl_WallMom_), nArr_wallMom(nArr_wallMom_),
        lower_limit(lower_limit_), upper_limit(upper_limit_), totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_),
        minusInvTau(minusInvTau_){}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
	const lattices::D3Q19GPUConstants c;
    Ind = Ind + lower_limit;

    if (Ind >= upper_limit)
      return;

    // Load the distribution functions
    // f[19] and fEq[19]
    double dev_ff[19]={0};   //, dev_fEq[19];
    double nn = 0.0;     // density
    double momentum_x, momentum_y, momentum_z;
    momentum_x = momentum_y = momentum_z = 0.0;

    double velx, vely, velz;   // Fluid Velocity


    //-----------------------------------------------------------------------------------------------------------
    // 1. Read the fOld_GPU_b distr. functions
    // 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
    // 		a. Calculate density
    // 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
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

    //--------------------------------------------------------------------------------------------------
    // c. Calculate equilibrium distr. functions
    double density_1 = 1.0 / nn;
    double momentumMagnitudeSquared = momentum_x * momentum_x + momentum_y * momentum_y 
						+ momentum_z * momentum_z;

    for (int i = 0; i < c.NUMVECTORS; ++i) {
      double mom_dot_ei = (double) c.CX[i] * momentum_x + (double) c.CY[i] * momentum_y 
			+ (double) c.CZ[i] * momentum_z;


		double dev_fEq = c.EQMWEIGHTS[i]
									* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
														+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
      dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
    }

    //----------------------------------------------------------------------------------------------------

    // d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
    // To do!!!
    //------------------------------------------------------------------------------------------------------

    // --------------------------------------------------------------------------------
    // Streaming Step:
    // 	a. Load the streaming indices
    // 	b. The Iolet-Fluid links info
    /**
                    If the link is an Iolet-Fluid link
                    c. Load the wallMom array
                    d. Compute the correction to the bounced back part of the distr. functions:
                    Hence, need to have the following:
                            d.1. LatticeType::c.EQMWEIGHTS[LB_dir]
                            d.2. LatticeType::c.CX[LB_dir], LatticeType::c.CY[LB_dir], LatticeType::c.CZ[LB_dir],
                            d.3. Cs2
                            d.4. Bounced-back index: just the c.INVERSEDIRECTIONS is sufficient
    */

    //
    // b. Iolet-Fluid links info:
    uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

    // Put the new populations after collision in the GMem_dbl array,
    // implementing the streaming step with Simple Bounce Back if Wall-Fluid link

    // fNew (dev_fn) populations:
    for (int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++) {
      unsigned mask = 1U << (LB_Dir - 1);   // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do:
                                            // compare against test_bool_Wall_Intersect as well)
      bool is_Iolet_link = (Iolet_Intersect & mask);

      if (is_Iolet_link) {   // ioletLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
        //===================================================================================================
        // c. Load the WallMom info - Note: We follow Method b for the data layout
        site_t siteCount = upper_limit - lower_limit;
        site_t shifted_Fluid_Ind = Ind - lower_limit;

        //-----------------------
        // Approach 2
        // July 2022 - Single value correction term (wall momentum) passed to the GPU global memory
        distribn_t correction = GMem_dbl_WallMom[(unsigned long long) (LB_Dir - 1) * siteCount + shifted_Fluid_Ind];

        // TODO: Pass the boolean variable: CollisionType::CKernel::LatticeType::IsLatticeCompressible()
        // Remember that the wall mom. does not include the correction (multiplication by local density) If Compressible:
        correction *= nn;
        //-----------------------

        int unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];

        GMem_dbl_fNew_b[(unsigned long long) unstreamed_dir * nArr_dbl + Ind] = dev_ff[LB_Dir] - correction;

        //==================================================================================================
      } else {   // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);

        // Use the Neighbouring Index given in GPUDataAddr_int64_Neigh_d, which is the actual streaming Array Index in f_new global memory
        int64_t dev_NeighInd = GMem_int64_Neigh[(unsigned long long) LB_Dir * nArr_dbl + Ind];

        // Save the post collision population in fNew
        GMem_dbl_fNew_b[dev_NeighInd] = dev_ff[LB_Dir];

        //---------------------------------------------------------------------------
      }   // Closes the bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
    }
    //=============================================================================================

    // Write old density and velocity to memory -
    if (write_GlobalMem) {
      GMem_dbl_MacroVars[Ind] = nn;
      GMem_dbl_MacroVars[1ULL * nArr_dbl + Ind] = velx;
      GMem_dbl_MacroVars[2ULL * nArr_dbl + Ind] = vely;
      GMem_dbl_MacroVars[3ULL * nArr_dbl + Ind] = velz;
    }

  }   // Ends the kernel GPU_Collide Type 3-4: mInletCollision - mOutletCollision & Velocity BCs
  //==========================================================================================
};   // End of Functor

//==========================================================================================
} // namespace hemelb

#include "./repro3_data_inlet2.h"

namespace hemelb { 

template<typename LatticeType>
bool setup()
{
	bool initialise_GPU_res = true;
	return initialise_GPU_res;

}
}; // end of namespace hemelb
