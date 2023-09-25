#include "cuda_kernels_def_decl/deviceAPI.h"
#include "lb/lattices/D3Q19_gpu.h"

namespace hemelb {

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
	const lb::lattices::D3Q19GPUConstants c;
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

	// To get around ROCM Compiler bugs 'portably'....
	GPU_DUMMY_SYNC;

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

  GPU_KERNEL void operator()(unsigned long long Ind) const {
	const lb::lattices::D3Q19GPUConstants c;

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

#ifndef HEMELB_USE_SYCL
    if (IdInlet == INT32_MAX) {
      printf("Fluid_ID : %ld, ID_iolet: %ld - Fluid NOT in IOLET range (NashZerothOrderPressure) !!! PID: %d file: lb.hpp line: %d  \n\n", Ind, IdInlet, myPiD, line);
	}
#endif

    ghost_dens = GMem_ghostDensity[IdInlet];
    inletNormal_x = GMem_inletNormal[3 * IdInlet];
    inletNormal_y = GMem_inletNormal[3 * IdInlet + 1];
    inletNormal_z = GMem_inletNormal[3 * IdInlet + 2];

    // Put the new populations after collision in the GMem_dbl array,
    // implementing the streaming step with Simple Bounce Back if Wall-Fluid link

    // fNew (dev_fn) populations:
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

//**************************************************************
// Kernel for the Collision step for the Lattice Boltzmann algorithm
// 		Pressure BCs: Option NASHZEROTHORDERPRESSUREIOLET
// 	Collision Types 3-4: mInletCollision & mOutletCollision: Inlet - Outlet BCs
//												Iolet's info accessed from GPU global memory
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
template <typename LatticeType> struct GPU_CollideStream_Iolets_NashZerothOrderPressure_v2_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Iolet_Link;
  distribn_t *GMem_ghostDensity;
  float *GMem_inletNormal;
  int nInlets;
  int64_t nArr_dbl;
  uint64_t lower_limit;
  uint64_t upper_limit;
  uint64_t totalSharedFs;
  bool write_GlobalMem;
  int num_local_Iolets;
  site_t *GMem_Iolets_info;
  double minusInvTau;


  GPU_CollideStream_Iolets_NashZerothOrderPressure_v2_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
                                                              int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Iolet_Link_, distribn_t *GMem_ghostDensity_,
                                                              float *GMem_inletNormal_, int nInlets_, uint64_t nArr_dbl_, uint64_t lower_limit_,
                                                              uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_, int num_local_Iolets_,
                                                              site_t *GMem_Iolets_info_, double minusInvTau_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), GMem_ghostDensity(GMem_ghostDensity_), GMem_inletNormal(GMem_inletNormal_), nInlets(nInlets_),
        nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_), totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_),
        num_local_Iolets(num_local_Iolets_), GMem_Iolets_info(GMem_Iolets_info_), minusInvTau(minusInvTau_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
	const lb::lattices::D3Q19GPUConstants c;
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
    velz = momentum_z / nn;
    vely = momentum_y / nn;

    //-------------------------------------------------------------------------------------------------------
    // c. Calculate equilibrium distr. functions
    double density_1 = 1.0 / nn;
    double momentumMagnitudeSquared = momentum_x * momentum_x 
			+ momentum_y * momentum_y + momentum_z * momentum_z;

    for (int i = 0; i < c.NUMVECTORS; ++i) {
      double mom_dot_ei = (double)c.CX[i] * momentum_x + (double) c.CY[i] * momentum_y + c.CZ[i] * momentum_z;

      double dev_fEq = c.EQMWEIGHTS[i] * (nn - (3.0 / 2.0) * (momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z) * density_1 +
                        (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

      dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
    }
    //-----------------------------------------------------------------------------------------------------------

    // d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
    // To do!!!
    //-----------------------------------------------------------------------------------------------------------

    // Collision step:
    // Single Relaxation Time approximation (LBGK)
    // double dev_fn[19];		// or maybe use the existing dev_ff[c.NUMVECTORS] to minimise the memory requirements - Check and replace in
    // the future


    // --------------------------------------------------------------------------------
    // Streaming Step:
    // a. Load the streaming indices
    // b. The Iolet-Fluid links info
    // c. The ghost density
    // d. The inletNormal

    // a. Bulk Streaming indices: dev_NeighInd[19] here refers to the ACTUAL Streaming Array index (Data Address) in f_old and f_new
    
    distribn_t ghost_dens;   // = 0.0; //new distribn_t[nInlets];	// c. The ghost density
    float inletNormal_x, inletNormal_y, inletNormal_z;

    //
    // b. Iolet-Fluid links info:
    uint32_t Iolet_Intersect = GMem_uint32_Iolet_Link[Ind];

    //--------------------------------------------------------------------------
    // Read the ghost density and the inlet Normal
    // Distinguish which inlet ID ???

    // Read the Iolet info (iolet ids and fluid sites range) from GMem_Iolets_info
    //	extern __shared__ int s[];
    // TODO: Consider using shared memory in the future as this info is read from all the threads

    // Identify the local iolet ID
    //  		There are 2 possible ways:
    // 			1. Using the information from GPU global mem (GMem_Iolets_info)
    //			2. Using struct Iolets containing the info (when number of iolets less than 30)

    int IdInlet = INT32_MAX;   // Iolet (Inlet/Outlet) ID
    if (num_local_Iolets == 1) {
      // Approach 1 - from GPU global mem (GMem_Iolets_info)
      IdInlet = (int)GMem_Iolets_info[0];
      // Approach 2 - from struct array
      // IdInlet = Iolets_info.Iolets_ID_range[0];
    } else {
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

#ifndef HEMELB_USE_SYCL
    // Debugging:
    if (IdInlet == INT32_MAX) {
      printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! FAILURE!!! Needs to abort...(NashZerothOrderPressure v2)\n\n", Ind, IdInlet);
    }
    //--------------------------------------------------------------------------
#endif

    ghost_dens = GMem_ghostDensity[IdInlet];
    inletNormal_x = GMem_inletNormal[3 * IdInlet];
    inletNormal_y = GMem_inletNormal[3 * IdInlet + 1];
    inletNormal_z = GMem_inletNormal[3 * IdInlet + 2];

// Put the new populations after collision in the GMem_dbl array,
// implementing the streaming step with Simple Bounce Back if Wall-Fluid link

// fNew (dev_fn) populations:
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

        //------------------------------------------------------------------------------------------------------
        // Calculate Feq[unstreamed_dir] - Only the direction that is necessary
        density_1 = 1.0 / ghost_dens;
        momentumMagnitudeSquared = momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z;

        int unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];
        double mom_dot_ei = (double) c.CX[unstreamed_dir] * momentum_x 
				+ (double) c.CY[unstreamed_dir] * momentum_y 
				+ (double) c.CZ[unstreamed_dir] * momentum_z;

        double dev_fEq_unstr = c.EQMWEIGHTS[unstreamed_dir] 
			* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1 
				+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
        //----------------------------------------------------------------------------------------------------
        // Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
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
    // Maybe use a different cuda kernel for these calculations (if saving the MacroVariables delays the collision/streaming kernel)
    // Check -  To do!!!
    if (write_GlobalMem) {
      GMem_dbl_MacroVars[Ind] = nn;
      GMem_dbl_MacroVars[1ULL * nArr_dbl + Ind] = velx;
      GMem_dbl_MacroVars[2ULL * nArr_dbl + Ind] = vely;
      GMem_dbl_MacroVars[3ULL * nArr_dbl + Ind] = velz;
    }

  }   // namespace hemelb
  //==========================================================================================
};   // End of functor
}   // namespace hemelb
