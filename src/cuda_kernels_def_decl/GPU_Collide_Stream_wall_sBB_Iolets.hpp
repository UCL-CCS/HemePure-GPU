#include "cuda_kernels_def_decl/deviceAPI.h"
#include "lb/lattices/D3Q19_gpu.h"

namespace hemelb {

template <typename LatticeType> struct GPU_CollideStream_wall_sBB_iolet_Nash_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Wall_Link;
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
  double minusInvTau;

  GPU_CollideStream_wall_sBB_iolet_Nash_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
                                                int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Wall_Link_, uint32_t *GMem_uint32_Iolet_Link_,
                                                distribn_t *GMem_ghostDensity_, float *GMem_inletNormal_, int nInlets_, uint64_t nArr_dbl_,
                                                uint64_t lower_limit_, uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_,
                                                int num_local_Iolets_, Iolets Iolets_info_, double minusInvTau_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_), GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), GMem_ghostDensity(GMem_ghostDensity_),
        GMem_inletNormal(GMem_inletNormal_), nInlets(nInlets_), nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_),
        totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_), num_local_Iolets(num_local_Iolets_),        Iolets_info(Iolets_info_), minusInvTau(minusInvTau_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
		const lb::lattices::D3Q19GPUConstants c;

		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

		double nn = (double)0; // density
		double velx=(double)0; // Fluid Velocities
		double vely=(double)0;
		double velz=(double)0;
		double momentum_x=(double)0;
		double momentum_y=(double)0;
		double momentum_z=(double)0;
		double dev_ff[19];



		// Load the distribution functions
		//f[19] and fEq[19]

		//-----------------------------------------------------------------------------------------
		// 1. Read the fOld_GPU_b distr. functions
		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
		// 		a. Calculate density
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
#pragma unroll 19
		for(size_t direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[direction * nArr_dbl + Ind];
			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
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

#pragma unroll 19
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)c.CX[i] * momentum_x
						+ (double)c.CY[i] * momentum_y
						+ (double)c.CZ[i] * momentum_z;

			double dev_fEq = c.EQMWEIGHTS[i]
													* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

		 	dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

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

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet =(int) Iolets_info.Iolets_ID_range[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
		}

#ifndef HEMELB_USE_SYCL
		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/
#endif
		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
#pragma unroll 19
		for(int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++)
		{
			// Avoid undefined behaviour setting the mask...:
			// No other uses of LB_Dir - 1 as indexing,
			unsigned mask = (LB_Dir > 0) ? 1U << (LB_Dir - 1) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);
			bool is_Wall_link  = (Wall_Intersect & mask);


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

				size_t unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];
				double mom_dot_ei = (double)c.CX[unstreamed_dir] * momentum_x
						+ (double)c.CY[unstreamed_dir] * momentum_y
						+ (double)c.CZ[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = c.EQMWEIGHTS[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				GMem_dbl_fNew_b[unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				// Simple Bounce Back case:
				GMem_dbl_fNew_b[c.INVERSEDIRECTIONS[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir

			}
			else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//---------------------------------------------------------------------------
				// If we use the elements in GMem_int64_Neigh_d - then we access the memory address in fOld or fNew directly (Method B: data arranged by LB_Dir)..
				// Including the info for the totalSharedFs (propagate outside of the simulation domain).
				// (remember the memory layout in hemeLB is based on the site fluid index (Method A), i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				int64_t dev_NeighInd = GMem_int64_Neigh[(size_t)LB_Dir * nArr_dbl + Ind]; // Depends on which neigh array is loaded... Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!

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

};    // End functor

// July 2024
// Wall shear stress calculations
template <typename LatticeType> struct GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Wall_Link;
  uint32_t *GMem_uint32_Iolet_Link;
  distribn_t *GMem_ghostDensity;
  float *GMem_inletNormal;
  int nInlets;
  uint64_t nArr_dbl;
  uint64_t lower_limit;
  uint64_t upper_limit;
  uint64_t totalSharedFs;
  bool write_GlobalMem;
  distribn_t* GMem_dbl_WallShearStressMagn; // write
  distribn_t* GMem_dbl_WallNormal;          // read
  distribn_t iStressParameter;
  int num_local_Iolets;
  Iolets Iolets_info;
  double minusInvTau;

  GPU_CollideStream_wall_sBB_iolet_Nash_WallShearStress_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
                                                int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Wall_Link_, uint32_t *GMem_uint32_Iolet_Link_,
                                                distribn_t *GMem_ghostDensity_, float *GMem_inletNormal_, int nInlets_, uint64_t nArr_dbl_,
                                                uint64_t lower_limit_, uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_,
                                                distribn_t* GMem_dbl_WallShearStressMagn_, distribn_t* GMem_dbl_WallNormal_, distribn_t iStressParameter_,
                                                int num_local_Iolets_, Iolets Iolets_info_, double minusInvTau_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_), GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), GMem_ghostDensity(GMem_ghostDensity_),
        GMem_inletNormal(GMem_inletNormal_), nInlets(nInlets_), nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_),
        totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_),
        GMem_dbl_WallShearStressMagn(GMem_dbl_WallShearStressMagn_), GMem_dbl_WallNormal(GMem_dbl_WallNormal_), iStressParameter(iStressParameter_),
        num_local_Iolets(num_local_Iolets_),        Iolets_info(Iolets_info_), minusInvTau(minusInvTau_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
		const lb::lattices::D3Q19GPUConstants c;

		Ind = Ind + lower_limit;

		if(Ind >= upper_limit)
			return;

		double nn = (double)0; // density
		double velx=(double)0; // Fluid Velocities
		double vely=(double)0;
		double velz=(double)0;
		double momentum_x=(double)0;
		double momentum_y=(double)0;
		double momentum_z=(double)0;
		double dev_ff[19];

		// Load the distribution functions
		//f[19] and fEq[19]

		//-----------------------------------------------------------------------------------------
		// 1. Read the fOld_GPU_b distr. functions
		// 2. Calculate the nessessary elements for calculating the equilibrium distribution functions
		// 		a. Calculate density
		// 		b. Calculate momentum - Needs to consider the case of body force as well - To do!!!
#pragma unroll 19
		for(size_t direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[direction * nArr_dbl + Ind];
			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
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

    double f_neq[19];
#pragma unroll 19
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)c.CX[i] * momentum_x
						+ (double)c.CY[i] * momentum_y
						+ (double)c.CZ[i] * momentum_z;

			double dev_fEq = c.EQMWEIGHTS[i]
													* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

      f_neq[i] = dev_ff[i] - dev_fEq;
		 	dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
      // Maybe switch to
      // dev_ff[i] += f_neq[i]  * minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

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

		// Determine the IdInlet - Done!!!
		int IdInlet = INT32_MAX; // Iolet (Inlet/Outlet) ID
		if(num_local_Iolets==1){
			IdInlet =(int) Iolets_info.Iolets_ID_range[0];//IdInlet = iolets_ID_range[0];
		}
		else{
			// Call a device function to determine which is the Iolet ID - using the iolets_ID_range Array
			// iolets_ID_range Array:
			//	a. Size: num_local_Iolets * 3
			// 	b. Iolet ID, Range of fluid IDs: [lower_limit, upper_limit]
			_determine_Iolet_ID(num_local_Iolets, Iolets_info.Iolets_ID_range, Ind, &IdInlet);
		}

#ifndef HEMELB_USE_SYCL
		// Testing:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! \n\n", Ind, IdInlet);
		}
/*		else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}
*/
#endif
		ghost_dens = GMem_ghostDensity[IdInlet];
		inletNormal_x = GMem_inletNormal[3*IdInlet];
		inletNormal_y = GMem_inletNormal[3*IdInlet+1];
		inletNormal_z = GMem_inletNormal[3*IdInlet+2];
		//printf("ghost_dens[%d]: %.5f, inletNormal_x = %.5f, inletNormal_y = %.5f, inletNormal_z = %.5f  \n\n", IdInlet, ghost_dens, inletNormal_x, inletNormal_y, inletNormal_z);

		// b. Wall-Fluid links info:
		uint32_t Wall_Intersect = GMem_uint32_Wall_Link[Ind];

		// Put the new populations after collision in the GMem_dbl array,
		// implementing the streaming step with:
		// Iolet BCs: NashZerothOrderPressure if iolet-fluid link
		// Wall BCs: Simple Bounce Back if wall-fluid link

		// fNew (dev_fn) populations:
#pragma unroll 19
		for(int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++)
		{
			// Avoid undefined behaviour setting the mask...:
			// No other uses of LB_Dir - 1 as indexing,
			unsigned mask = (LB_Dir > 0) ? 1U << (LB_Dir - 1) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);
			bool is_Wall_link  = (Wall_Intersect & mask);


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

				size_t unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];
				double mom_dot_ei = (double)c.CX[unstreamed_dir] * momentum_x
						+ (double)c.CY[unstreamed_dir] * momentum_y
						+ (double)c.CZ[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = c.EQMWEIGHTS[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				GMem_dbl_fNew_b[unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

			}
			else if(is_Wall_link){	// wallLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				// Simple Bounce Back case:
				GMem_dbl_fNew_b[c.INVERSEDIRECTIONS[LB_Dir] * nArr_dbl + Ind]= dev_ff[LB_Dir]; // Bounce Back - Same fluid ID - Reverse LB_Dir

			}
			else{ // bulkLinkDelegate.StreamLink(lbmParams, latDat, site, hydroVars, ii);
				//---------------------------------------------------------------------------
				// If we use the elements in GMem_int64_Neigh_d - then we access the memory address in fOld or fNew directly (Method B: data arranged by LB_Dir)..
				// Including the info for the totalSharedFs (propagate outside of the simulation domain).
				// (remember the memory layout in hemeLB is based on the site fluid index (Method A), i.e. f0[0], f1[0], f2[0], ..., fq[0] and for the Fluid Index Ind : f0[Ind], f1[Ind], f2[Ind], ..., fq[Ind]

				int64_t dev_NeighInd = GMem_int64_Neigh[(size_t)LB_Dir * nArr_dbl + Ind]; // Depends on which neigh array is loaded... Read the streaming info here - Here Refers to Data Address NOT THE STREAMING FLUID ID!!!

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
						iStressParameter);
      //printf("(3) Wall Shear Stress = %5.5e\n", stress );
      GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			//------------------------------------------------------------------------

		}
	} // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
	//==========================================================================================

};    // End functor

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
template <typename LatticeType> struct GPU_CollideStream_wall_sBB_iolet_Nash_v2_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Wall_Link;
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
  site_t *GMem_Iolets_info;
  double minusInvTau;

  GPU_CollideStream_wall_sBB_iolet_Nash_v2_Functor(distribn_t *GMem_dbl_fOld_b_, distribn_t *GMem_dbl_fNew_b_, distribn_t *GMem_dbl_MacroVars_,
                                                   int64_t *GMem_int64_Neigh_, uint32_t *GMem_uint32_Wall_Link_, uint32_t *GMem_uint32_Iolet_Link_,
                                                   distribn_t *GMem_ghostDensity_, float *GMem_inletNormal_, int nInlets_, uint64_t nArr_dbl_,
                                                   uint64_t lower_limit_, uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_,
                                                   int num_local_Iolets_, site_t *GMem_Iolets_info_, double minusInvTau_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_), GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), GMem_ghostDensity(GMem_ghostDensity_),
        GMem_inletNormal(GMem_inletNormal_), nInlets(nInlets_), nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_),
        totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_), num_local_Iolets(num_local_Iolets_), GMem_Iolets_info(GMem_Iolets_info_), minusInvTau(minusInvTau_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
		const lb::lattices::D3Q19GPUConstants c;

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
		for(int direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
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
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)c.CX[i] * momentum_x
							+ (double)c.CY[i] * momentum_y
							+ (double)c.CZ[i] * momentum_z;

		  double dev_fEq = c.EQMWEIGHTS[i]
													* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

			dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[c.NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		/*// Evolution equation for the fi's here
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * minusInvTau; // Check if multiplying by minusInvTau makes a difference
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
			IdInlet =(int) GMem_Iolets_info[0];
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

#ifndef HEMELB_USE_SYCL
		// Debugging:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! FAILURE!!! Needs to abort... \n\n", Ind, IdInlet);
		}
		/*else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}*/
		//--------------------------------------------------------------------------
#endif

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
#pragma unroll 19
		for(int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++) {

			// Avoid UB in the shift operator for a -ve number{
			unsigned mask =  LB_Dir > 0 ? 1U << (LB_Dir - 1) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);
			bool is_Wall_link = (Wall_Intersect & mask);


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

				int unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];
				double mom_dot_ei = (double)c.CX[unstreamed_dir] * momentum_x
							+ (double)c.CY[unstreamed_dir] * momentum_y
							+ (double)c.CZ[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = c.EQMWEIGHTS[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * c.NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

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
		//if(time_Step%_Send_MacroVars_DtH == 0){
		if (write_GlobalMem){
			GMem_dbl_MacroVars[Ind] = nn;
			GMem_dbl_MacroVars[1ULL*nArr_dbl + Ind] = velx;
			GMem_dbl_MacroVars[2ULL*nArr_dbl + Ind] = vely;
			GMem_dbl_MacroVars[3ULL*nArr_dbl + Ind] = velz;
		}

  }   // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
      //==========================================================================================
};    // end functor


template <typename LatticeType> struct GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress_Functor {

  distribn_t *GMem_dbl_fOld_b;
  distribn_t *GMem_dbl_fNew_b;
  distribn_t *GMem_dbl_MacroVars;
  int64_t *GMem_int64_Neigh;
  uint32_t *GMem_uint32_Wall_Link;
  uint32_t *GMem_uint32_Iolet_Link;
  distribn_t *GMem_ghostDensity;
  float *GMem_inletNormal;
  int nInlets;
  uint64_t nArr_dbl;
  uint64_t lower_limit;
  uint64_t upper_limit;
  uint64_t totalSharedFs;
  bool write_GlobalMem;
  distribn_t* GMem_dbl_WallShearStressMagn; // write
  distribn_t* GMem_dbl_WallNormal;          // read
  distribn_t iStressParameter;
  int num_local_Iolets;
  site_t *GMem_Iolets_info;
  double minusInvTau;

  GPU_CollideStream_wall_sBB_iolet_Nash_v2_WallShearStress_Functor(distribn_t *GMem_dbl_fOld_b_,
    distribn_t *GMem_dbl_fNew_b_,
    distribn_t *GMem_dbl_MacroVars_,
    int64_t *GMem_int64_Neigh_,
    uint32_t *GMem_uint32_Wall_Link_,
    uint32_t *GMem_uint32_Iolet_Link_,
    distribn_t *GMem_ghostDensity_,
    float *GMem_inletNormal_,
    int nInlets_,
    uint64_t nArr_dbl_,
    uint64_t lower_limit_,
    uint64_t upper_limit_,
    uint64_t totalSharedFs_,
    bool write_GlobalMem_,
    distribn_t* GMem_dbl_WallShearStressMagn_, distribn_t* GMem_dbl_WallNormal_, distribn_t iStressParameter_,
    int num_local_Iolets_, site_t *GMem_Iolets_info_, double minusInvTau_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_), GMem_dbl_fNew_b(GMem_dbl_fNew_b_), GMem_dbl_MacroVars(GMem_dbl_MacroVars_), GMem_int64_Neigh(GMem_int64_Neigh_),
        GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_), GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_), GMem_ghostDensity(GMem_ghostDensity_),
        GMem_inletNormal(GMem_inletNormal_), nInlets(nInlets_), nArr_dbl(nArr_dbl_), lower_limit(lower_limit_), upper_limit(upper_limit_),
        totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_),
        GMem_dbl_WallShearStressMagn(GMem_dbl_WallShearStressMagn_), GMem_dbl_WallNormal(GMem_dbl_WallNormal_), iStressParameter(iStressParameter_),
        num_local_Iolets(num_local_Iolets_), GMem_Iolets_info(GMem_Iolets_info_), minusInvTau(minusInvTau_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
		const lb::lattices::D3Q19GPUConstants c;

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
		for(int direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
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
#pragma unroll 19
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)c.CX[i] * momentum_x
							+ (double)c.CY[i] * momentum_y
							+ (double)c.CZ[i] * momentum_z;

		  double dev_fEq = c.EQMWEIGHTS[i]
													* (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
																	+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

      f_neq[i] = dev_ff[i] - dev_fEq;
      dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
      // Maybe switch to
      // dev_ff[i] += f_neq[i]  * minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------

		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

		// Collision step:
		// Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[c.NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		/*// Evolution equation for the fi's here
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			//dev_fn[i] = dev_ff[i] + (dev_fEq[i] - dev_ff[i])/dev_tau; // + force[i];
			dev_ff[i] += (dev_ff[i] - dev_fEq[i]) * minusInvTau; // Check if multiplying by minusInvTau makes a difference
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
			IdInlet =(int) GMem_Iolets_info[0];
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

#ifndef HEMELB_USE_SYCL
		// Debugging:
		if(IdInlet==INT32_MAX)
		{
			printf("Fluid_ID : %lld, ID_iolet: %d - Fluid NOT in IOLET range!!! FAILURE!!! Needs to abort... \n\n", Ind, IdInlet);
		}
		/*else{
			printf("Fluid_ID : %lld, ID_iolet: %d \n\n", Ind, IdInlet);
		}*/
		//--------------------------------------------------------------------------
#endif

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
#pragma unroll 19
		for(int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++) {

			// Avoid UB in the shift operator for a -ve number{
			unsigned mask =  LB_Dir > 0 ? 1U << (LB_Dir - 1) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);
			bool is_Wall_link = (Wall_Intersect & mask);


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

				int unstreamed_dir = c.INVERSEDIRECTIONS[LB_Dir];
				double mom_dot_ei = (double)c.CX[unstreamed_dir] * momentum_x
							+ (double)c.CY[unstreamed_dir] * momentum_y
							+ (double)c.CZ[unstreamed_dir] * momentum_z;

				double dev_fEq_unstr = c.EQMWEIGHTS[unstreamed_dir]
							* (ghost_dens - (3.0 / 2.0) * momentumMagnitudeSquared * density_1
											+ (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);
				//------------------------------------------------------------------------------------------------------
				// Need to distinguish the int boundaryId = site.GetIoletId() correctly and pass the info (identify the proper ghost density and inlet-normals.
				//=============================================================================================================

				// printf("Site ID = %lld - Inlet in Dir: %d, Unstreamed direction: %d, fEq = %.5e \n\n", Ind, LB_Dir, unstreamed_dir, dev_fEq_unstr);

				// Case of NashZerothOrderPressure:
				// *latticeData->GetFNew(site.GetIndex() * c.NUMVECTORS + unstreamed) = ghostHydrovars.GetFEq()[unstreamed];
				GMem_dbl_fNew_b[(unsigned long long)unstreamed_dir * nArr_dbl + Ind] = dev_fEq_unstr;

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
						iStressParameter);
      //printf("(3) Wall Shear Stress = %5.5e\n", stress );
      GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			//------------------------------------------------------------------------
		}

  }   // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
      //==========================================================================================
};    // end functor


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

  GPU_KERNEL void operator()(unsigned long long Ind) const {
	const lb::lattices::D3Q19GPUConstants c;
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
#pragma unroll 19
		for(int direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
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
		for(int i = 0; i < c.NUMVECTORS; ++i)
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
#pragma unroll 19
		for(int i = 0; i < c.NUMVECTORS; ++i)
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
#pragma unroll 19
		for(int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++)
		{
			// Avoid UB here
			unsigned mask = LB_Dir > 0 ? 1U << (LB_Dir - 1) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);
			bool is_Wall_link = (Wall_Intersect & mask);

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

// July 2024
//  Wall shear stress calculations added to the functorised version
//**************************************************************
template <typename LatticeType> struct GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress_Functor {
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
  distribn_t* GMem_dbl_WallShearStressMagn; // write
  distribn_t* GMem_dbl_WallNormal;          // read
  distribn_t iStressParameter;
  const double minusInvTau;
  const double Cs2;

  GPU_CollideStream_wall_sBB_Iolets_Ladd_VelBCs_WallShearStress_Functor(
      distribn_t *GMem_dbl_fOld_b_,
      distribn_t *GMem_dbl_fNew_b_,
      distribn_t *GMem_dbl_MacroVars_,
      int64_t *GMem_int64_Neigh_,
      uint32_t *GMem_uint32_Wall_Link_,
      uint32_t *GMem_uint32_Iolet_Link_,
      uint64_t nArr_dbl_,
      distribn_t *GMem_dbl_WallMom_,
      uint64_t nArr_wallMom_, uint64_t lower_limit_,
      uint64_t upper_limit_, uint64_t totalSharedFs_, bool write_GlobalMem_,
      distribn_t* GMem_dbl_WallShearStressMagn_, distribn_t* GMem_dbl_WallNormal_, distribn_t iStressParameter_,
      double minusInvTau_, double Cs2_)
      : GMem_dbl_fOld_b(GMem_dbl_fOld_b_),
      GMem_dbl_fNew_b(GMem_dbl_fNew_b_),
      GMem_dbl_MacroVars(GMem_dbl_MacroVars_),
      GMem_int64_Neigh(GMem_int64_Neigh_),
      GMem_uint32_Wall_Link(GMem_uint32_Wall_Link_),
      GMem_uint32_Iolet_Link(GMem_uint32_Iolet_Link_),
      nArr_dbl(nArr_dbl_),
      GMem_dbl_WallMom(GMem_dbl_WallMom_),
      nArr_wallMom(nArr_wallMom_),
      lower_limit(lower_limit_), upper_limit(upper_limit_), totalSharedFs(totalSharedFs_), write_GlobalMem(write_GlobalMem_),
      GMem_dbl_WallShearStressMagn(GMem_dbl_WallShearStressMagn_), GMem_dbl_WallNormal(GMem_dbl_WallNormal_), iStressParameter(iStressParameter_),
      minusInvTau(minusInvTau_), Cs2(Cs2_) {}

  GPU_KERNEL void operator()(unsigned long long Ind) const {
	const lb::lattices::D3Q19GPUConstants c;
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
		for(int direction = 0; direction< c.NUMVECTORS; direction++){
			dev_ff[direction] = GMem_dbl_fOld_b[(unsigned long long)direction * nArr_dbl + Ind];

			nn += dev_ff[direction];
			momentum_x += (double)c.CX[direction] * dev_ff[direction];
			momentum_y += (double)c.CY[direction] * dev_ff[direction];
			momentum_z += (double)c.CZ[direction] * dev_ff[direction];
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
    // and
    // Collision step - Evolution equation for the fi's
    // Single Relaxation Time approximation (LBGK)
		//double dev_fn[19];		// or maybe use the existing dev_ff[c.NUMVECTORS] to minimise the memory requirements - Check and replace in the future

		double density_1 = 1.0 / nn;

    double f_neq[19];
#pragma unroll 19
		for(int i = 0; i < c.NUMVECTORS; ++i)
		{
			double mom_dot_ei = (double)c.CX[i] * momentum_x
					+ (double)c.CY[i] * momentum_y
					+ (double)c.CZ[i] * momentum_z;

      // c. Calculate equilibrium distr. functions
      double dev_fEq = c.EQMWEIGHTS[i]
                        * (nn - (3.0 / 2.0) * ( momentum_x * momentum_x + momentum_y * momentum_y + momentum_z * momentum_z ) * density_1
                        + (9.0 / 2.0) * density_1 * mom_dot_ei * mom_dot_ei + 3.0 * mom_dot_ei);

      f_neq[i] = dev_ff[i] - dev_fEq;

      // Evolution equation for the fi's here
      dev_ff[i] += (dev_ff[i] - dev_fEq) * minusInvTau;
		}
		//-----------------------------------------------------------------------------------------------------------
		// d. Body Force case: Add details of any forcing scheme here - Evaluate force[i]
		// To do!!!
		//-----------------------------------------------------------------------------------------------------------

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
#pragma unroll 19
		for(int LB_Dir = 0; LB_Dir < c.NUMVECTORS; LB_Dir++)
		{
			// Avoid UB here
			unsigned mask = LB_Dir > 0 ? 1U << (LB_Dir - 1) : 0; // Needs to left shift the bits in mask so that I can then compare against the value in test_Wall_Intersect (To do: compare against test_bool_Wall_Intersect as well)
			bool is_Iolet_link = (Iolet_Intersect & mask);
			bool is_Wall_link = (Wall_Intersect & mask);

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

		// Write
    // A. density, velocity
    // B. Wall shear stress magnitude
    // Consider having 2 separate boolean vars for these (e.g. write_GlobalMem_dens_vel and write_GlobalMem_wallShearStress)
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
        iStressParameter);

			//printf("(3) Wall Shear Stress = %5.5e\n", stress );
			GMem_dbl_WallShearStressMagn[shifted_Ind] = stress;
			//------------------------------------------------------------------------

		}
  }   // Ends the kernel GPU_Collide Type 6: Outlets-Wall - PreReceive
      //==========================================================================================
};    // End functor





};    // namespace hemelb
